#!/usr/bin/env bash
#
# Repeat a single-split baseline-vs-CLAHE comparison across several seeds.
#
#   bash scripts/run_seeds.sh 43 44                    # leaked ids excluded
#   bash scripts/run_seeds.sh --include-leaked 42 43 44 # reproduce the history
#
# Measures how much of an apparent difference between two variants is real and
# how much is run-to-run noise.
#
# Leakage. The six single-split runs recorded in RESULTS.md were made with an
# earlier version of this script that did NOT exclude the 49 training images
# duplicated in valid or test, while run_cv.sh always has. That made the two
# sets of results incomparable on leakage handling without saying so. Exclusion
# is now the default, matching cross-validation; --include-leaked reproduces the
# historical runs exactly. (Measured: excluding them moved test QWK from 0.8960
# to 0.8983 on seed 42, so the difference is small - but it is not nothing, and
# it should be a choice rather than an accident.)
#
set -uo pipefail

cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore

LEAK_FLAG="--exclude-leaked"
if [[ "${1:-}" == "--include-leaked" ]]; then
    LEAK_FLAG=""
    shift
fi

if (( $# )); then
    SEEDS=("$@")
else
    SEEDS=(43 44)
fi

FAILED=()
for seed in "${SEEDS[@]}"; do
    for variant in baseline clahe; do
        if [[ "$variant" == "clahe" ]]; then data_dir="data/processed_clahe"; else data_dir="data/processed"; fi

        echo "=================================================="
        echo "variant=$variant  seed=$seed  leaks=${LEAK_FLAG:-included}"
        echo "=================================================="

        # The old version printed "(exit $?)" after an echo, which reported the
        # status of the echo rather than of python.
        if python -u scripts/train.py \
            --mode reg --model efficientnet_b0 --size 384 --batch 16 \
            --epochs 15 --patience 5 --lr 3e-4 --workers 2 \
            --seed "$seed" --data-dir "$data_dir" --variant "$variant" \
            --no-bq $LEAK_FLAG; then
            echo "--- $variant seed $seed: ok"
        else
            code=$?
            echo "--- $variant seed $seed: FAILED (exit $code)"
            FAILED+=("$variant/$seed")
        fi
    done
done

if (( ${#FAILED[@]} )); then
    echo "failed runs: ${FAILED[*]}"
    exit 1
fi
echo "all seed runs completed"
