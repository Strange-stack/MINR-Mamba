import argparse
import datetime
import logging
import math
import os
import random
import time
from os import path as osp

import numpy as np
import torch

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           make_exp_dirs, mkdir_and_rename, set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='Distributed launcher.')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()
    seed = opt.get('manual_seed') or random.randint(1, 10000)
    opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])
    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    train_loader, val_loader = None, None
    total_epochs, total_iters = 0, int(opt['train']['total_iter'])

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(
                train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'],
            )

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_epochs = math.ceil(total_iters / num_iter_per_epoch)
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size: {opt["world_size"]}'
                f'\n\tIterations per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; total iters: {total_iters}.')
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'],
            )
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def find_latest_resume_state(opt):
    state_folder_path = osp.join('experiments', opt['name'], 'training_states')
    try:
        states = os.listdir(state_folder_path)
    except FileNotFoundError:
        return None
    states = [x for x in states if x.endswith('.state')]
    if not states:
        return None
    max_state_file = f'{max(int(x[:-6]) for x in states if x.endswith(".state"))}.state'
    return osp.join(state_folder_path, max_state_file)


def load_resume_state(opt):
    resume_state_path = find_latest_resume_state(opt)
    if resume_state_path is not None:
        opt['path']['resume_state'] = resume_state_path

    if not opt['path'].get('resume_state'):
        return None

    device_id = torch.cuda.current_device()
    return torch.load(
        opt['path']['resume_state'],
        map_location=lambda storage, loc: storage.cuda(device_id),
    )


def setup_prefetcher(opt, train_loader, logger):
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        return CPUPrefetcher(train_loader)
    if prefetch_mode == 'cuda':
        logger.info('Use cuda prefetch dataloader.')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
        return CUDAPrefetcher(train_loader, opt)
    raise ValueError(
        f"Wrong prefetch_mode {prefetch_mode}. Supported values are None, 'cpu', and 'cuda'.")


def prepare_progressive_schedule(opt):
    train_opt = opt['datasets']['train']
    iters = train_opt.get('iters')
    groups = np.array([sum(iters[0:i + 1]) for i in range(len(iters))])
    return {
        'batch_size': train_opt.get('batch_size_per_gpu'),
        'mini_batch_sizes': train_opt.get('mini_batch_sizes'),
        'gt_size': train_opt.get('gt_size'),
        'mini_gt_sizes': train_opt.get('gt_sizes'),
        'groups': groups,
        'logged': [True] * len(groups),
        'scale': opt['scale'],
    }


def apply_progressive_training(train_data, current_iter, schedule, logger):
    unfinished = ((current_iter > schedule['groups']) != True).nonzero()[0]
    stage_idx = len(schedule['groups']) - 1 if len(unfinished) == 0 else unfinished[0]
    mini_gt_size = schedule['mini_gt_sizes'][stage_idx]
    mini_batch_size = schedule['mini_batch_sizes'][stage_idx]

    if schedule['logged'][stage_idx]:
        logger.info(
            f'\nUpdating patch size to {mini_gt_size} and batch size to '
            f'{mini_batch_size * torch.cuda.device_count()}\n')
        schedule['logged'][stage_idx] = False

    lq = train_data['lq']
    gt = train_data['gt']
    if mini_batch_size < schedule['batch_size']:
        indices = random.sample(range(schedule['batch_size']), k=mini_batch_size)
        lq = lq[indices]
        gt = gt[indices]

    if mini_gt_size < schedule['gt_size']:
        x0 = int((schedule['gt_size'] - mini_gt_size) * random.random())
        y0 = int((schedule['gt_size'] - mini_gt_size) * random.random())
        x1 = x0 + mini_gt_size
        y1 = y0 + mini_gt_size
        scale = schedule['scale']
        lq = lq[:, :, x0:x1, y0:y1]
        gt = gt[:, :, x0 * scale:x1 * scale, y0 * scale:y1 * scale]

    return {'lq': lq, 'gt': gt}


def main():
    opt = parse_options(is_train=True)
    torch.backends.cudnn.benchmark = True

    resume_state = load_resume_state(opt)
    if resume_state is None:
        make_exp_dirs(opt)
        if (opt['logger'].get('use_tb_logger')
                and 'debug' not in opt['name']
                and opt['rank'] == 0):
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    logger, tb_logger = init_loggers(opt)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = (
        create_train_val_dataloader(opt, logger))

    if resume_state:
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)
        logger.info(
            f"Resuming training from epoch: {resume_state['epoch']}, "
            f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0

    msg_logger = MessageLogger(opt, current_iter, tb_logger)
    prefetcher = setup_prefetcher(opt, train_loader, logger)
    schedule = prepare_progressive_schedule(opt)

    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()
    epoch = start_epoch

    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_time = time.time() - data_time
            current_iter += 1
            if current_iter > total_iters:
                break

            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            train_data = apply_progressive_training(
                train_data, current_iter, schedule, logger)
            model.feed_train_data(train_data)
            model.optimize_parameters(current_iter)

            iter_time = time.time() - iter_time
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            if opt.get('val') is not None and current_iter % opt['val']['val_freq'] == 0:
                model.validation(
                    val_loader,
                    current_iter,
                    tb_logger,
                    opt['val']['save_img'],
                    opt['val'].get('rgb2bgr', True),
                    opt['val'].get('use_image', True),
                )

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        epoch += 1

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)
    if opt.get('val') is not None:
        model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    main()
