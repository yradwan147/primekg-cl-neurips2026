#!/bin/bash
# =============================================================
# Run 15: Additional decoder experiments (RotatE, ComplEx)
# Purpose: Strengthen "decoder choice matters" finding in Paper A
# Methods: Naive Sequential + EWC with RotatE and ComplEx
# Total: 4 methods × 5 seeds = 20 jobs
# =============================================================

set -e
mkdir -p slurm/slurm_logs

SEEDS="42 123 456 789 1024"
DECODERS="RotatE ComplEx"
METHODS="naive_sequential ewc"

echo "=== Run 15: RotatE + ComplEx decoder experiments ==="
echo "Jobs: 4 methods × 5 seeds = 20 jobs"
echo ""

for DECODER in $DECODERS; do
  for METHOD in $METHODS; do
    for SEED in $SEEDS; do
      # Create short job name
      if [ "$METHOD" == "naive_sequential" ]; then
        SHORT="ns"
      else
        SHORT="$METHOD"
      fi
      DSHORT=$(echo $DECODER | tr '[:upper:]' '[:lower:]' | cut -c1-3)
      JOBNAME="${SHORT}_${DSHORT}_s${SEED}"

      echo "Submitting: $METHOD $DECODER seed=$SEED -> $JOBNAME"
      sbatch -J "$JOBNAME" slurm/run_baseline.sh "$METHOD" "$DECODER" "$SEED"
    done
  done
done

echo ""
echo "=== All 20 jobs submitted ==="
echo "Monitor: squeue -u \$USER --format='%.10i %.14j %.8T %.10M'"
echo "Results will appear in: results/{method}_{Decoder}_seed{N}.json"
