# MINR-Mamba

MINR-Mamba is an image deraining project for paired rainy/clean image
restoration. The repository contains a training and inference pipeline, a
MINR-Mamba network definition, and common research utilities for losses,
schedulers, and restoration metrics.

The model combines a multi-scale encoder-decoder restoration backbone with
attentive state-space blocks and two coordinate-MLP preprocessing branches. One
branch operates on the HSV value channel, and the other operates on RGB
features.

## Repository Structure

```text
.
├── Deraining/
│   ├── Options/MINR_Mamba.yml      # default training and validation config
│   ├── test.py                     # inference script
│   └── utils.py                    # image loading and saving helpers
├── basicsr/
│   ├── data/                       # paired LQ/GT dataset and dataloader
│   ├── metrics/                    # PSNR, SSIM, and NIQE metrics
│   ├── models/                     # model wrapper, losses, schedulers, network
│   ├── train.py                    # training entry point
│   └── utils/                      # logging, options, image and file helpers
├── requirements.txt
├── requirements-mamba.txt          # CUDA extension dependencies
├── environment.yml                 # optional Conda environment
├── THIRD_PARTY_NOTICES.md          # upstream code and license notes
├── CITATION.cff                    # citation metadata
├── setup.py
└── train.sh
```

## Version

Current project version: `0.1.0`.

The version number is stored in `VERSION`, `basicsr/version.py`, and
`setup.py`.

## Environment Setup

MINR-Mamba depends on `mamba-ssm`, which builds CUDA/PyTorch extensions. Install
PyTorch first, then install the ordinary Python dependencies, and install the
Mamba packages last.

Create the environment:

```bash
cd MINR-Mamba
conda create -n minr-mamba python=3.10 -y
conda activate minr-mamba
```

Alternatively, create the base Conda environment from the provided file:

```bash
conda env create -f environment.yml
conda activate minr-mamba
```

Install a CUDA-compatible PyTorch build. Choose the command that matches your
GPU driver and CUDA runtime. For example, for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Install the common dependencies:

```bash
pip install -r requirements.txt
```

Install the Mamba CUDA extension dependencies:

```bash
pip install packaging ninja
pip install -r requirements-mamba.txt --no-build-isolation
```

Install this repository in editable mode:

```bash
pip install -e .
```

Verify the environment:

```bash
python - <<'PY'
import torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('selective_scan_fn:', selective_scan_fn is not None)
PY
```

If `mamba-ssm` installation fails, check that the active PyTorch build, CUDA
runtime, compiler, and `causal-conv1d` wheel/source build are compatible. The
network imports `selective_scan_fn`, so this verification step must pass before
training or inference.

## Dataset Layout

The default config expects paired rainy inputs and clean targets:

```text
datasets/
├── Training/
│   ├── LQ/
│   └── GT/
└── Testing/
    ├── LQ/
    └── GT/
```

The LQ and GT filenames must match after applying `filename_tmpl` in
`Deraining/Options/MINR_Mamba.yml`. To use another dataset location, edit:

```yaml
dataroot_lq: /path/to/rainy/images
dataroot_gt: /path/to/clean/images
```

The dataset directories are intentionally ignored by git and are not part of
this repository.

## Training

Single-GPU training:

```bash
cd MINR-Mamba
bash train.sh
```

Equivalent direct command:

```bash
python basicsr/train.py -opt Deraining/Options/MINR_Mamba.yml --launcher none
```

Multi-GPU training:

```bash
NUM_GPUS=8 bash train.sh Deraining/Options/MINR_Mamba.yml
```

Training outputs are written under:

```text
experiments/MINR_Mamba/
├── models/
├── training_states/
├── visualization/
└── train_MINR_Mamba_*.log
```

If `experiments/MINR_Mamba/training_states/` contains `.state` files, training
resumes from the latest saved iteration.

## Inference

Run deraining on a folder of input images:

```bash
cd MINR-Mamba
python Deraining/test.py \
  --input_dir /path/to/rainy/images \
  --result_dir ./results \
  --weights /path/to/checkpoint.pth
```

Pretrained checkpoints are not included in this repository. The `--weights`
argument is required and must point to a trained MINR-Mamba checkpoint.

The inference script pads each image to a multiple of 8 with reflection padding,
runs the network in evaluation mode, crops the output back to the original
resolution, clamps values to `[0, 1]`, and writes PNG results.

## Configuration

The main configuration file is `Deraining/Options/MINR_Mamba.yml`.

Important fields:

- `network_g.type`: network class name, currently `MINRMamba`.
- `datasets.train` and `datasets.val`: paired image dataset settings.
- `train.total_iter`: total training iterations.
- `train.scheduler`: learning rate scheduler settings.
- `val.metrics`: validation metric settings. PSNR is enabled by default, while
  SSIM and NIQE implementations are also available.
- `logger`: print, checkpoint, and TensorBoard logging intervals.

The network implementation is in `basicsr/models/archs/minr_mamba_arch.py`.
The original Transformer-only architecture files are intentionally not included
in this cleaned MINR-Mamba package.

## Git Hygiene

The `.gitignore` excludes datasets, checkpoints, experiment outputs, TensorBoard
logs, generated results, archives, and Python cache files. Keep datasets and
`.pth` checkpoints outside git.

## License And Attribution

This project keeps code derived from existing image restoration projects. Review
`LICENSE.md` and `THIRD_PARTY_NOTICES.md`, and keep the necessary upstream
license notices and citations when publishing or distributing the code.
