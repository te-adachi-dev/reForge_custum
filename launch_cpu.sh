#!/bin/bash
source reforge_env/bin/activate
export CUDA_VISIBLE_DEVICES=""
python launch.py \
    --skip-torch-cuda-test \
    --precision full \
    --no-half \
    --use-cpu all \
    --always-cpu \
    --listen \
    --port 7860 \
    --lowram \
    --disable-nan-check
