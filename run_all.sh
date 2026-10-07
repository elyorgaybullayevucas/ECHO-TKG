#!/usr/bin/env bash
# Three seeds of the full model per dataset, one tmux session per GPU.
#   GPUS="0 1 2 3" ./run_all.sh
#   GPUS="2 2 1" ./run_all.sh          # two queues on GPU 2, one on GPU 1
#   GPUS="0 1" DATASETS="ICEWS18 YAGO" ./run_all.sh
# Then the ablations on one dataset:
#   GPUS="0 1 2 3" ABLATE=ICEWS18 ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
mkdir -p logs checkpoints
# tmux starts a fresh shell that may not have the conda env active, so pin
# the interpreter that is active right now.
PY="$(command -v python)"
read -ra G <<< "${GPUS:?set GPUS, e.g. GPUS=\"0 1\"}"
jobs=()
if [[ -n "${ABLATE:-}" ]]; then
  for flag in no_compete no_path no_proto no_ctx no_type no_stream no_dyad no_pop no_struct; do
    jobs+=("--dataset $ABLATE --seed 1 --tag s1 --$flag")
  done
else
  for ds in ${DATASETS:-ICEWS18 YAGO WIKI GDELT}; do
    for seed in 1 2 3; do jobs+=("--dataset $ds --seed $seed --tag s$seed"); done
  done
fi
# Each entry of GPUS is one queue; the same card may appear twice to run two
# queues on it. Jobs are dealt round-robin over the queues.
declare -A chain
for i in "${!jobs[@]}"; do
  q=$((i % ${#G[@]}))
  name=$(echo "${jobs[$i]}" | tr -s ' -' '_')
  chain[$q]+="$PY -u train_echo.py ${jobs[$i]} --gpu ${G[$q]} 2>&1 | tee logs/run${name}.out; "
done
for q in "${!G[@]}"; do
  [[ -n "${chain[$q]:-}" ]] || continue
  tmux new -d -s "echo_q${q}_gpu${G[$q]}" "cd $(pwd) && ${chain[$q]}"
  echo "queue $q on GPU ${G[$q]}: tmux session echo_q${q}_gpu${G[$q]}"
done
echo "results: python collect.py"
