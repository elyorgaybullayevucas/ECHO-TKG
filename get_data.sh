#!/usr/bin/env bash
# Fetch the standard benchmark splits (the RE-GCN / CEN files) into ./data.
#   ./get_data.sh                 # ICEWS14s ICEWS18 YAGO WIKI GDELT
#   ./get_data.sh ICEWS18 YAGO
set -euo pipefail
cd "$(dirname "$0")"
BASE="https://raw.githubusercontent.com/Lee-zix/CEN/main/data"
SETS=("$@"); [[ ${#SETS[@]} -eq 0 ]] && SETS=(ICEWS14s ICEWS18 YAGO WIKI GDELT)
for ds in "${SETS[@]}"; do
  mkdir -p "data/$ds"
  for f in train.txt valid.txt test.txt; do
    [[ -s "data/$ds/$f" ]] || curl -fL --retry 3 -o "data/$ds/$f" "$BASE/$ds/$f"
  done
  echo "$ds: $(cat data/$ds/{train,valid,test}.txt | wc -l) facts"
done
