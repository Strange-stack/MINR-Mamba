import importlib
import os
import random
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from os import path as osp

import torch
import torch.nn.functional as F

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class MixingAugment:
    """Optional mixup augmentation used during training."""

    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(
            torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device
        self.use_identity = use_identity
        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()
        r_index = torch.randperm(target.size(0)).to(self.device)
        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


class ImageCleanModel(BaseModel):
    """Image restoration model wrapper for training and validation."""

    def __init__(self, opt):
        super().__init__(opt)
        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = MixingAugment(
                mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        load_path = self.opt['path'].get('pretrain_network_g')
        if load_path is not None:
            self.load_network(
                self.net_g,
                load_path,
                self.opt['path'].get('strict_load_g', True),
                param_key=self.opt['path'].get('param_key', 'params'),
            )

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = define_network(deepcopy(self.opt['network_g'])).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g')
            if load_path is not None:
                self.load_network(
                    self.net_g_ema,
                    load_path,
                    self.opt['path'].get('strict_load_g', True),
                    'params_ema',
                )
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)
        else:
            raise ValueError('pixel loss is None.')

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for name, param in self.net_g.named_parameters():
            if param.requires_grad:
                optim_params.append(param)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {name} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(f'Optimizer {optim_type} is not supported.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]
        self.output = preds[-1]

        loss_dict = OrderedDict()
        l_pix = 0.
        for pred in preds:
            l_pix += self.cri_pix(pred, self.gt)
        loss_dict['l_pix'] = l_pix

        l_pix.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale,
                                  0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            self.output = pred[-1] if isinstance(pred, list) else pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            self.output = pred[-1] if isinstance(pred, list) else pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img,
                        rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(
                dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img,
                           rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        window_size = self.opt['val'].get('window_size', 0)
        test = partial(self.pad_test, window_size) if window_size else self.nonpad_test
        count = 0

        for val_data in dataloader:
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'],
                        img_name,
                        f'{img_name}_{current_iter}.png',
                    )
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'],
                        img_name,
                        f'{img_name}_{current_iter}_gt.png',
                    )
                else:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')
                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    if use_image:
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                    else:
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(
                                visuals['result'], visuals['gt'], **opt_)
            count += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= count
                current_metric = self.metric_results[metric]
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network(
                [self.net_g, self.net_g_ema],
                'net_g',
                current_iter,
                param_key=['params', 'params_ema'],
            )
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
