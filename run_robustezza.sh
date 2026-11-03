#!/bin/sh
# Ripartizioni temporali (test: giorni 1-10, 11-20, 21-30) e semi (0,1,2 sulla ripartizione 21-30). Default = test 21-30, seme 0.
export PYTHONWARNINGS=ignore
run() {  # citta tag testA testB seed
  echo "######## $1 $2 test=$3-$4 seed=$5"
  TAG=$2 TEST_A=$3 TEST_B=$4 SEED=$5 python3 src/build_kb.py $1 > /dev/null
  TAG=$2 TEST_A=$3 TEST_B=$4 SEED=$5 python3 src/hmm.py $1 | grep "K scelto"
  TAG=$2 TEST_A=$3 TEST_B=$4 SEED=$5 python3 src/hmm_classi.py $1 | grep -E "^\[|profilo p"
  TAG=$2 TEST_A=$3 TEST_B=$4 SEED=$5 python3 src/valutazione.py $1
  TAG=$2 TEST_A=$3 TEST_B=$4 SEED=$5 python3 src/valuta_csp.py $1
}
echo "######## dc default"; python3 src/valutazione.py dc
for c in jc bos dc; do
  run $c f1s0 1 10 0
  run $c f2s0 11 20 0
  run $c s1 21 30 1
  run $c s2 21 30 2
done
echo "######## FINE"
