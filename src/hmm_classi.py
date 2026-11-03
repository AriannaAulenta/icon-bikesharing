"""Un HMM per ogni profilo di stazione (classi della KB). Stessa struttura di hmm.py; K scelto con BIC per profilo."""
import sys, re, pickle, numpy as np
sys.path.insert(0, "src")
import hmm as H
from build_kb import atomo
from config import TRAIN_IDX, TEST_IDX, SEED, SUF

def etichette(c, st):
    lab = {}
    for riga in open(f"kb/generated/{c}{SUF}.pl"):
        mm = re.match(r"prop\((s_\w+), profilo, p(\d+)\)\.", riga)
        if mm: lab[mm.group(1)] = int(mm.group(2))
    return np.array([lab[atomo(s)] for s in st])

def main(c, Ks=(2, 3, 4, 5)):
    st, saldo, giorni, weekend = H.carica_seq(c); lab = etichette(c, st); modelli = {}; ll_h, ll_b = [], []
    Ote = H.simbolo(saldo[:, TEST_IDX, :])
    per_giorno = np.zeros((10, 2)); pesi = np.zeros(10)
    for p in sorted(set(lab)):
        sel = lab == p; Otr = H.simbolo(saldo[sel][:, TRAIN_IDX, :]).reshape(-1, H.T)
        res = {}
        for K in Ks:
            best = max((H.em(Otr, K, H.FAS_DI_ORA, iters=20, seed=SEED * 10 + s) for s in range(2)), key=lambda m: m["ll"])
            res[K] = (-2*best["ll"] + H.npar(K)*np.log(Otr.size), best)
        Kb = min(res, key=lambda k: res[k][0]); m = res[Kb][1]; modelli[p] = {k: m[k] for k in ("pi", "A", "B")}
        print(f"  profilo p{p}: stazioni={sel.sum()} K={Kb} (BIC per K: { {k: round(v[0]) for k, v in res.items()} })", flush=True)
        P = H.baseline_ora(Otr)
        for d in range(10):
            O = Ote[sel][:, d, :]; n = len(O)
            per_giorno[d] += n * np.array([H.log_loss_hmm(m, O).mean(), H.log_loss_base(P, O).mean()]); pesi[d] += n
    r = per_giorno / pesi[:, None]
    print(f"[{c}] log-loss a un passo giorni 21-30: HMM per profilo {r[:,0].mean():.4f}±{r[:,0].std():.4f} | baseline per profilo e ora {r[:,1].mean():.4f}±{r[:,1].std():.4f} | HMM migliore in {int((r[:,0]<r[:,1]).sum())}/10 giorni")
    pickle.dump({"lab": lab, "modelli": modelli}, open(f"data/proc/hmmc_{c}{SUF}.pkl", "wb"))

if __name__ == "__main__": main(sys.argv[1])
