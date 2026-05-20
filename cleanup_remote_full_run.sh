#!/usr/bin/env bash
set -euo pipefail

echo "[1/4] disk usage before cleanup"
df -h
echo

echo "[2/4] remove failed full-training outputs on system disk"
rm -rf /root/deadline520/LLaMA-Factory/output/qwen25_csqa_full
rm -f /root/deadline520/output/csqa_full_train_memory.json
rm -f /root/deadline520/output/csqa_full_eval.json
rm -f /root/deadline520/output/full_train.log
echo "done"
echo

echo "[3/4] create output directories on data disk"
mkdir -p /root/autodl-tmp/qwen25_csqa_full
mkdir -p /root/autodl-tmp/deadline520-output
echo "done"
echo

echo "[4/4] disk usage after cleanup"
df -h
echo
echo "Ready to rerun Full SFT."
echo "Training output dir: /root/autodl-tmp/qwen25_csqa_full"
echo "Monitor json dir:    /root/autodl-tmp/deadline520-output"
