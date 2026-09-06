from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (
    paired_paths_from_folder,
    paired_paths_from_meta_info_file,
)
from basicsr.data.transforms import paired_random_crop, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding


class Dataset_PairedImage(data.Dataset):
    """Paired LQ/GT image dataset for image restoration."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean')
        self.std = opt.get('std')
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        if self.io_backend_opt['type'] != 'disk':
            raise ValueError('Dataset_PairedImage now supports only io_backend: disk.')

        if opt.get('meta_info_file') is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt.get('geometric_augs', False)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)

        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except Exception as exc:
            raise RuntimeError(f'GT path {gt_path} cannot be read.') from exc

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except Exception as exc:
            raise RuntimeError(f'LQ path {lq_path} cannot be read.') from exc

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            img_gt, img_lq = paired_random_crop(
                img_gt, img_lq, gt_size, scale, gt_path)
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
        }

    def __len__(self):
        return len(self.paths)
