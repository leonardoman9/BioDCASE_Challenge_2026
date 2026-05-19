#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST_ROOT="${SCRIPT_DIR}/eval_data"
REMOTE_ROOT="/mnt/sda4/Progetti/BioDCASE_Challenge_2026/biodcase_model"
REMOTE_DATASET="${REMOTE_ROOT}/BioDCASE2026_TinyML_Development_Dataset/Validation"
REMOTE_CLASS_MAP="${REMOTE_ROOT}/outputs/class_map.json"

mkdir -p "${DEST_ROOT}/BioDCASE2026_TinyML_Development_Dataset"

echo "syncing Validation split from archleo..."
rsync -avz "archleo:${REMOTE_DATASET}/" "${DEST_ROOT}/BioDCASE2026_TinyML_Development_Dataset/Validation/"

echo "syncing class_map.json from archleo..."
rsync -avz "archleo:${REMOTE_CLASS_MAP}" "${DEST_ROOT}/class_map.json"

echo
echo "done:"
echo "  dataset:   ${DEST_ROOT}/BioDCASE2026_TinyML_Development_Dataset/Validation"
echo "  class_map: ${DEST_ROOT}/class_map.json"
