import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from natsort import natsorted
from tqdm import tqdm

import utils
from basicsr.models.archs.minr_mamba_arch import MINRMamba


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description='Image deraining with MINR-Mamba')
    parser.add_argument(
        '--input_dir',
        default=str(repo_root / 'datasets' / 'Testing' / 'LQ'),
        type=str,
        help='Directory containing rainy input images.')
    parser.add_argument(
        '--result_dir',
        default=str(repo_root / 'results'),
        type=str,
        help='Directory for restored images.')
    parser.add_argument(
        '--weights',
        required=True,
        type=str,
        help='Path to a MINR-Mamba checkpoint.')
    parser.add_argument(
        '--config',
        default=str(
            Path(__file__).resolve().parent / 'Options' / 'MINR_Mamba.yml'),
        type=str,
        help='Path to the network YAML configuration.')
    return parser.parse_args()


def load_network_options(config_path):
    with open(config_path, mode='r') as file:
        try:
            loader = yaml.CLoader
        except AttributeError:
            loader = yaml.Loader
        opt = yaml.load(file, Loader=loader)
    network_opt = opt['network_g']
    network_opt.pop('type', None)
    return network_opt


def img_as_uint8(img):
    """Convert a float image in [0, 1] to uint8."""
    return np.clip(img * 255.0, 0, 255).round().astype(np.uint8)


def main():
    args = parse_args()
    network_opt = load_network_options(args.config)

    model = MINRMamba(**network_opt)
    checkpoint = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(checkpoint['params'])
    print(f'===> Testing with weights: {args.weights}')

    model.cuda()
    model = nn.DataParallel(model)
    model.eval()

    os.makedirs(args.result_dir, exist_ok=True)
    files = natsorted(
        glob(os.path.join(args.input_dir, '*.png')) +
        glob(os.path.join(args.input_dir, '*.jpg')) +
        glob(os.path.join(args.input_dir, '*.jpeg'))
    )
    if not files:
        raise FileNotFoundError(f'No input images found in {args.input_dir}')

    factor = 8
    with torch.no_grad():
        for file_path in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(utils.load_img(file_path)) / 255.
            img = torch.from_numpy(img).permute(2, 0, 1)
            input_tensor = img.unsqueeze(0).cuda()

            height, width = input_tensor.shape[2], input_tensor.shape[3]
            padded_h = ((height + factor) // factor) * factor
            padded_w = ((width + factor) // factor) * factor
            pad_h = padded_h - height if height % factor != 0 else 0
            pad_w = padded_w - width if width % factor != 0 else 0
            input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), 'reflect')

            restored = model(input_tensor)
            restored = restored[:, :, :height, :width]
            restored = torch.clamp(restored, 0, 1)
            restored = (
                restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy())

            image_name = os.path.splitext(os.path.basename(file_path))[0] + '.png'
            result_path = os.path.join(args.result_dir, image_name)
            utils.save_img(result_path, img_as_uint8(restored))


if __name__ == '__main__':
    main()
