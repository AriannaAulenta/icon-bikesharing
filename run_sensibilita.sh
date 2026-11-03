#!/bin/sh
export PYTHONWARNINGS=ignore SOLO_KB=1
for c in jc bos dc; do
  for cfg in "r01 0.1 0.85 0.65" "r02 0.2 0.85 0.65" "r08 0.8 0.85 0.65" "q95 0.4 0.95 0.85" "q70 0.4 0.70 0.50"; do
    set -- $cfg
    echo "######## $c $1 raggio=$2 qalto=$3 qmedio=$4"
    TAG=$1 RAGGIO=$2 Q_ALTO=$3 Q_MEDIO=$4 python3 src/build_kb.py $c > /dev/null
    TAG=$1 RAGGIO=$2 Q_ALTO=$3 Q_MEDIO=$4 python3 src/valutazione.py $c
  done
done
echo "######## FINE"
