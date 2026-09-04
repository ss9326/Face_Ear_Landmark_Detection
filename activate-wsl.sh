#!/usr/bin/env bash

# Activate the project environment and expose pip-installed CUDA libraries.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"

NVIDIA_DIR="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia"
CUDA_LIBRARY_PATH=""
for library_dir in "$NVIDIA_DIR"/*/lib; do
    if [[ -d "$library_dir" ]]; then
        CUDA_LIBRARY_PATH="${CUDA_LIBRARY_PATH:+$CUDA_LIBRARY_PATH:}$library_dir"
    fi
done
export LD_LIBRARY_PATH="${CUDA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset NVIDIA_DIR CUDA_LIBRARY_PATH library_dir

echo "Activated $VIRTUAL_ENV"
