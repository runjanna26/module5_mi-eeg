#!/bin/bash
# Train the named HWNet variants (see configs/HWNet.py) in parallel.
# Usage: ./run_variants.sh <variant> [<variant> ...]
# Results go to logs_tuning/HWNet/<run>_<variant>/.
mkdir -p run_logs
for v in "$@"; do
    ( HW_VARIANT="$v" python run_HWNet.py --log_dir logs_tuning > "run_logs/tune_$v.log" 2>&1
      echo "$v exit $?" ) &
done
wait
