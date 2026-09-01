#!/usr/bin/env bash
# The implementation sweep on one dedicated cluster GPU, so the rows compare.
set -euo pipefail
Q=${1:-a100}
DEST=${DEST:-/groups/saalfeld/home/allierc/GraphData/ngp-demo-bench}
SSH=${SSH:-allierc@login1}
ENV=${ENV:-connectome-gnn}
CFGS=${CFGS:-"zapbench_bench zapbench_bench_compile zapbench_bench_bf16 zapbench_bench_warp zapbench_bench_warp_sadam"}
rsync -a --delete --exclude '.git' --exclude 'out/' --exclude 'log/' \
      --exclude '__pycache__' --exclude 'papers/' ./ "$DEST"/
JOB="cd $DEST"
for c in $CFGS; do JOB="$JOB && python NGP_Main.py -o bench $c"; done
ssh "$SSH" "bash -lc 'mkdir -p $DEST/log/zapbench_bench && cd $DEST && bsub -n 8 \
  -gpu num=1 -q gpu_$Q -W 120 -oo $DEST/log/zapbench_bench/sweep_$Q.lsf.log \
  -J ngpsweep_$Q bash -lc \"conda run --no-capture-output -n $ENV bash -c \\\"$JOB\\\"\"'"
echo "collect: cp $DEST/log/zapbench_bench/bench_*.json log/zapbench_bench/ && python NGP_Main.py -o table zapbench_bench"
