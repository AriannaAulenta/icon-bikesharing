#!/bin/sh
export PYTHONWARNINGS=ignore
python3 src/build_kb.py
for c in jc bos dc; do
  echo "######## $c"; python3 src/hmm.py $c; python3 src/hmm_classi.py $c; python3 src/valutazione.py $c; python3 src/valuta_csp.py $c
done
echo "######## FINE"
