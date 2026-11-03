"""Valutazione del solo CSP: rispetto dei vincoli e qualita' dell'obiettivo rispetto a un'euristica golosa (stessi punteggi: deficit reale)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
import hmm as H, ridistribuzione as R
from config import TEST_IDX, SEED
FIN, M = 3, 40
def main(c):
    st, saldo, giorni, weekend = H.carica_seq(c)
    ana = pd.read_csv(f"data/proc/{c}_stazioni.csv", dtype={"stazione": str}).set_index("stazione").loc[st]
    lat, lng = ana.lat.values, ana.lng.values; rng = np.random.RandomState(SEED); viol, rel, vinc = [], [], 0; n = 0
    for d in TEST_IDX:
        for t0 in (7, 11, 16):
            s = np.maximum(0.0, -saldo[:, d, t0:t0+FIN].sum(1)); pool = np.argsort(-(s + 1e-6*rng.rand(len(s))))[:M]
            D = R.haversine(lat[pool], lng[pool]); sp = s[pool]
            x, _ = R.risolvi_csp(sp, D, rng); xg = R.goloso(sp, D)
            viol.append(R.violazioni(x, D)); a, b = sp[x > 0].sum(), sp[xg > 0].sum()
            if b > 0: rel.append(a / b)
            n += 1
    rel = np.array(rel)
    print(f"[{c}] {n} istanze | violazioni dei vincoli nella soluzione: {int(np.sum(viol))} (istanze con violazioni: {int(np.sum(np.array(viol)>0))}) | obiettivo ricerca locale / goloso = {rel.mean():.3f} ± {rel.std():.3f} (ricerca locale >= goloso in {int((rel>=0.999).sum())}/{len(rel)})", flush=True)
if __name__ == "__main__":
    for c in sys.argv[1:]: main(c)
