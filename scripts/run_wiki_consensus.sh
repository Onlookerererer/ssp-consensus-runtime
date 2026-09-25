#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
seed="${1:-123}"
mkdir -p results/partial_labels results/consensus
# Stage existing candidate caches without overwriting server caches.
for cache in partial_labels/wiki_*_partial_labels_len_5.mat; do
    [[ -f "$cache" ]] || continue
    destination="results/partial_labels/${cache##*/}"
    if [[ ! -e "$destination" ]]; then
        cp -- "$cache" "$destination"
    fi
done
# Dataset settings follow the existing launcher and its parser defaults.
for partial_length in 5; do
    run_dir="results/consensus/wiki_L${partial_length}_seed${seed}_$(date +%Y%m%d_%H%M%S_%N)"
    mkdir -- "$run_dir"
    "${PYTHON:-python}" -B -u train.py \
        --dataset wiki --partial_length "$partial_length" --seed "$seed" \
        --MAX_EPOCH 180 --batch_size 2048 --output_dim 1024 \
        --lr 1e-4 --lamda 0.1 --ema_decay 0.95 \
        --neighbor_mode consensus --independent_train_seed \
        --neighbor_k 10 --neighbor_beta 0.2 \
        --neighbor_margin 0.10 --neighbor_support_threshold 0.20 \
        --log_dir "$run_dir" --best_checkpoint "$run_dir/best_validation.pt"
done
