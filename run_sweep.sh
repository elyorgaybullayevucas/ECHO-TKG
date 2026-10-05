#!/usr/bin/env bash
# Regularisation and capacity sweep on one dataset, selected on VALIDATION.
#
#   GPUS="5" ./run_sweep.sh                    # ICEWS18, one queue on GPU 5
#   GPUS="5 0" PER_GPU=2 ./run_sweep.sh        # two queues on each card
#   GPUS="5" DATASET=ICEWS14s ./run_sweep.sh
#
# Why: every run so far peaked at epoch 8-12 of 40 while the training loss
# kept falling, with the learning rate still near its maximum. The model
# overfits before the cosine schedule has annealed at all. Each row below
# changes one thing against the same base, so the table reads as an
# ablation of the training recipe.
#
# The last row trains the structural branch alone. It is a diagnostic, not a
# candidate: it says how strong the backbone is without any history lookup.
#
# Results:  python collect.py     (pick the row with the best "valid MRR")
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
mkdir -p logs checkpoints
# tmux starts a fresh shell that may not have the conda env active, so pin
# the interpreter that is active right now.
PY="$(command -v python)"
DS="${DATASET:-ICEWS18}"
PER_GPU="${PER_GPU:-1}"
read -ra G <<< "${GPUS:?set GPUS, e.g. GPUS=\"5\"}"
BASE="--dataset $DS --seed 42 --no_compete"

jobs=(
  "e16|--epochs 16"
  "e16d35|--epochs 16 --dropout 0.35"
  "e16d45|--epochs 16 --dropout 0.45"
  "e16wd|--epochs 16 --weight_decay 1e-3"
  "e20lr5|--epochs 20 --lr 5e-4"
  "e16aux1|--epochs 16 --struct_aux 1.0"
  "e16aux0|--epochs 16 --struct_aux 0.0"
  "e16ls2|--epochs 16 --label_smoothing 0.2"
  "e16wide|--epochs 16 --dyad_support 128 --path_support 32"
  "structonly|--epochs 30 --no_dyad --no_pop"
)

queues=()
for g in "${G[@]}"; do for ((k = 0; k < PER_GPU; k++)); do queues+=("$g"); done; done
declare -A chain
for i in "${!jobs[@]}"; do
  q=$((i % ${#queues[@]}))
  tag="${jobs[$i]%%|*}"; args="${jobs[$i]#*|}"
  chain[$q]+="$PY -u train_echo.py $BASE $args --tag $tag --device cuda --gpu ${queues[$q]} 2>&1 | tee logs/sweep_${DS}_${tag}.out; "
done
for q in "${!queues[@]}"; do
  [[ -n "${chain[$q]:-}" ]] || continue
  tmux new -d -s "sweep_${DS}_q$q" "cd $(pwd) && ${chain[$q]}"
  echo "queue $q on GPU ${queues[$q]}: tmux session sweep_${DS}_q$q"
done
echo "results: python collect.py"
