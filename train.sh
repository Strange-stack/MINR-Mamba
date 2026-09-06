#!/usr/bin/env bash

set -euo pipefail

CONFIG=${1:-Deraining/Options/MINR_Mamba.yml}
NUM_GPUS=${NUM_GPUS:-1}
MASTER_PORT=${MASTER_PORT:-4321}

if [ "$NUM_GPUS" -eq 1 ]; then
  python basicsr/train.py -opt "$CONFIG" --launcher none
else
  torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" basicsr/train.py -opt "$CONFIG" --launcher pytorch
fi
