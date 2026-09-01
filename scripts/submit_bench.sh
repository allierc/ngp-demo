#!/usr/bin/env bash
# Submit the NGP benchmark to the LSF GPU queues, one job per card.
#
#   bash scripts/submit_bench.sh                      # l4 a100 h100
#   bash scripts/submit_bench.sh a100 h100
#
# LSF lives on the login node, not in the devcontainer, so every call goes
# through `ssh <user>@<login> bash -lc`, exactly as connectome-gnn's runners do.
# The repo and its data are staged onto /groups, which both sides mount at the
# same path -- so the jobs write their bench_<gpu>.json where the container can
# read them, and `NGP_Main.py -o table` collects every card into one table.
set -euo pipefail
CFG=${CFG:-zapbench_bench}
DEST=${DEST:-/groups/saalfeld/home/allierc/GraphData/ngp-demo-bench}
SSH=${SSH:-allierc@login1}
ENV=${ENV:-connectome-gnn}
QUEUES=${*:-l4 a100 h100}

rsync -a --delete --exclude '.git' --exclude 'out/' --exclude 'log/' \
      --exclude '__pycache__' --exclude 'papers/' ./ "$DEST"/
echo "staged $(du -sh "$DEST" | cut -f1) at $DEST"

for q in $QUEUES; do
  ssh "$SSH" "bash -lc 'mkdir -p $DEST/log/$CFG && cd $DEST && bsub -n 8 -gpu num=1 \
    -q gpu_$q -W 60 -oo $DEST/log/$CFG/bench_$q.lsf.log -J ngpbench_$q \
    bash -lc \"cd $DEST && conda run --no-capture-output -n $ENV \
    python NGP_Main.py -o bench $CFG\"'"
done

echo
echo "watch:   ssh $SSH \"bash -lc 'bjobs'\""
echo "collect: cp $DEST/log/$CFG/bench_*.json log/$CFG/ && python NGP_Main.py -o table $CFG"
