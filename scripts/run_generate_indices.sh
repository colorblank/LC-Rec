#!/bin/bash

# 设置默认参数
DATASET="Games"
CKPT_PATH="/path/to/checkpoint"
OUTPUT_DIR="/path/to/output"
DEVICE="cuda:0"

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET="$2"
      shift
      shift
      ;;
    --ckpt_path)
      CKPT_PATH="$2"
      shift
      shift
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    --device)
      DEVICE="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown parameter passed: $1"
      exit 1
      ;;
  esac
done

# 执行 Python 脚本
python3 -m index/generate_indices.py \
  --dataset "$DATASET" \
  --ckpt_path "$CKPT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE"