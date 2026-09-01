#!/usr/bin/env bash
# One L4 job per NGP setting, all scored against the same injected stripes.
#
# Sixteen settings sweeping four knobs one at a time about the baseline
# (L=16, T=2^22, 2 px, 2 frames), which is the fit that scored 13.193 where a
# median filter scored worse than doing nothing.  Each job is a couple of
# minutes on an L4 with the warp stack, and they do not talk to each other, so
# they go out as sixteen jobs rather than one loop.
set -euo pipefail
Q=${Q:-l4}
DEST=${DEST:-/groups/saalfeld/home/allierc/GraphData/ngp-demo-bench}
SSH=${SSH:-allierc@login1}
ENV=${ENV:-connectome-gnn}
MODES=${MODES:-"shot:0.02 stripe:0.1 stripe:0.8"}
CONFIG=${CONFIG:-bisons}
STEPS=${STEPS:-1500}

# L  log2T px frames axis
SETTINGS="
16 22 2 2 base
16 18 2 2 log2T
16 20 2 2 log2T
16 21 2 2 log2T
16 23 2 2 log2T
6  22 2 2 levels
8  22 2 2 levels
12 22 2 2 levels
20 22 2 2 levels
16 22 1 2 px
16 22 4 2 px
16 22 8 2 px
16 22 2 1 frames
16 22 2 4 frames
16 22 2 8 frames
16 22 2 16 frames
"

rsync -a --delete --exclude '.git' --exclude 'out/' --exclude 'log/' \
      --exclude '__pycache__' --exclude 'papers/' ./ "$DEST"/

n=0
while read -r L T PX FR AXIS; do
  [ -z "$L" ] && continue
  n=$((n + 3))
  for M in $MODES; do
    SLUG=$(echo "$M" | tr -d ':.')
    CMD="python scripts/sweep_inject.py --config $CONFIG --mode $M --levels $L --log2t $T --px $PX --frames $FR --axis $AXIS --steps $STEPS"
    ssh -n "$SSH" "bash -lc 'mkdir -p $DEST/log/sweep_inject && cd $DEST && bsub -n 4 \
      -gpu num=1 -q gpu_$Q -W 60 -oo $DEST/log/sweep_inject/${SLUG}_L${L}_T${T}_px${PX}_f${FR}.lsf.log \
      -J ngpsw_${SLUG}_${L}_${T}_${PX}_${FR} bash -lc \"conda run --no-capture-output -n $ENV $CMD\"'" \
      | tail -1
  done
done <<< "$SETTINGS"

echo "$n jobs on gpu_$Q"
echo "collect: cp $DEST/log/sweep_inject/*.json log/sweep_inject/ && python scripts/sweep_inject.py --table"
