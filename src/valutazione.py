"""Valutazione end-to-end sui giorni 21-30 (mai visti da KB e HMM): quanto squilibrio reale riesce a coprire ogni politica,
a parita' di vincoli (CSP + A*). Metrica: deficit reale servito / deficit reale servito dall'oracolo."""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "src")
import os, pickle, hmm as H, ridistribuzione as R
from config import TRAIN_IDX, TEST_IDX, SEED, SUF
from build_kb import atomo
from kb import KB

T0S = {7: "mattina", 11: "giorno", 16: "sera"}     # istanti di decisione e fascia corrispondente
FINESTRA, M = 3, 40                                 # ore previste, stazioni candidate

def previsione_hmm(m, saldo, d, t0, rep):
    O = H.simbolo(saldo[:, d, :t0])
    alpha, _, _ = H.fb(O, m["pi"], m["A"], m["B"], H.FAS_DI_ORA[:t0], backward=False)
    b = alpha[:, -1]; tot = np.zeros(len(b))
    for h in range(t0, t0 + FINESTRA):
        b = b @ m["A"]; tot += b @ (m["B"][H.FAS_DI_ORA[h]] @ rep)
    return tot

def previsione_classi(mc, saldo, d, t0, rep):
    out = np.zeros(saldo.shape[0])
    for p, mod in mc["modelli"].items():
        sel = mc["lab"] == p
        out[sel] = previsione_hmm(mod, saldo[sel], d, t0, rep)
    return out

def main(c, seed=SEED):
    st, saldo, giorni, weekend = H.carica_seq(c)
    ana = pd.read_csv(f"data/proc/{c}_stazioni.csv", dtype={"stazione": str}).set_index("stazione").loc[st]
    lat, lng = ana.lat.values, ana.lng.values; depot = np.array([lat.mean(), lng.mean()])
    LEGGERA = bool(os.environ.get("SOLO_KB"))
    if not LEGGERA: z = np.load(f"data/proc/hmm_{c}{SUF}.npz"); m = {k: z[k] for k in ("pi", "A", "B")}
    Otr = H.simbolo(saldo[:, TRAIN_IDX, :]); rep = np.array([saldo[:, TRAIN_IDX, :][Otr == s].mean() for s in range(H.NSIM)])
    if not LEGGERA: mc = pickle.load(open(f"data/proc/hmmc_{c}{SUF}.pkl", "rb"))
    kb = KB(c, SUF); atomi = [atomo(s) for s in st]; idx = {a: i for i, a in enumerate(atomi)}
    cache = {}
    def kb_scores(tp, fa):
        if (tp, fa) not in cache:
            def punt(q):
                v = np.zeros(len(st))
                for r in kb.q(q): v[idx[r["S"]]] = max(v[idx[r["S"]]], r["P"])
                return v
            pr = punt(f"priorita(S,{tp},{fa},P)"); comp = np.zeros(len(st), bool)
            for r in kb.q(f"compensata(S,{tp},{fa})"): comp[idx[r["S"]]] = True
            cache[(tp, fa)] = (pr, comp, punt(f"priorita_senza_compensazione(S,{tp},{fa},P)"), punt(f"priorita_senza_pesi(S,{tp},{fa},P)"), punt(f"priorita_solo_saldo(S,{tp},{fa},P)"))
        return cache[(tp, fa)]
    rng = np.random.RandomState(seed); righe = []; astar = []
    for d in TEST_IDX:
        tp = "festivo" if weekend[d] else "feriale"
        for t0, fa in T0S.items():
            reale = np.maximum(0.0, -saldo[:, d, t0:t0+FINESTRA].sum(1))
            pr, comp, pr_nc, pr_np, pr_ss = kb_scores(tp, fa)
            hm = hc = np.zeros(len(st)) if LEGGERA else None
            if not LEGGERA: hm = np.maximum(0.0, -previsione_hmm(m, saldo, d, t0, rep)); hc = np.maximum(0.0, -previsione_classi(mc, saldo, d, t0, rep))
            punteggi = {"casuale": rng.rand(len(st)), "persistenza": np.maximum(0.0, -saldo[:, d, t0-FINESTRA:t0].sum(1)),
                        "KB (regole)": pr, "HMM globale": hm, "HMM per profilo (KB)": hc, "oracolo": reale,
                        "KB senza compensazione": pr_nc, "KB senza pesi": pr_np, "solo saldo medio": pr_ss}
            if LEGGERA: punteggi = {k: punteggi[k] for k in ("KB (regole)", "oracolo", "KB senza compensazione", "solo saldo medio")}
            res = {}
            for nome, s in punteggi.items():
                pool = np.argsort(-(s + 1e-6 * rng.rand(len(s))))[:M]
                D = R.haversine(lat[pool], lng[pool]); x, _ = R.risolvi_csp(s[pool], D, rng)
                servite = pool[x > 0]; res[nome] = reale[servite].sum()
                if nome in ("HMM per profilo (KB)", "oracolo"):
                    for v in range(1, R.V + 1):
                        sel = pool[x == v]
                        if len(sel):
                            a_, ga, ea = R.percorso_astar(depot, np.column_stack([lat[sel], lng[sel]]), True)
                            _, gu, eu = R.percorso_astar(depot, np.column_stack([lat[sel], lng[sel]]), False)
                            astar.append((len(sel), ga, ea, eu, abs(ga - gu) < 1e-9))
            if res["oracolo"] > 0:
                righe.append({"giorno": d + 1, "t0": t0, **{k: v / res["oracolo"] for k, v in res.items() if k != "oracolo"}})
    df = pd.DataFrame(righe); df.to_csv(f"data/proc/valutazione_{c}{SUF}.csv", index=False)
    pol = ["KB (regole)", "KB senza compensazione", "solo saldo medio"] if LEGGERA else ["casuale", "persistenza", "KB (regole)", "HMM globale", "HMM per profilo (KB)", "KB senza compensazione", "KB senza pesi", "solo saldo medio"]
    print(f"\n[{c}] scenari validi: {len(df)}")
    for p in pol: print(f"  {p:14s} copertura/oracolo = {df[p].mean():.3f} ± {df[p].std():.3f}")
    if LEGGERA: return
    h = "HMM per profilo (KB)"
    print(f"  KB (regole) > solo saldo medio in {(df['KB (regole)'] > df['solo saldo medio']).sum()}/{len(df)}; > senza compensazione in {(df['KB (regole)'] > df['KB senza compensazione']).sum()}/{len(df)}; > senza pesi in {(df['KB (regole)'] > df['KB senza pesi']).sum()}/{len(df)}")
    print(f"  {h} > KB in {(df[h] > df['KB (regole)']).sum()}/{len(df)} scenari; > persistenza in {(df[h] > df['persistenza']).sum()}/{len(df)}; > casuale in {(df[h] > df['casuale']).sum()}/{len(df)}; > HMM globale in {(df[h] > df['HMM globale']).sum()}/{len(df)}")
    a = np.array(astar, float)
    print(f"  A*: {len(a)} percorsi, fermate medie {a[:,0].mean():.1f}, tempo di viaggio medio {a[:,1].mean():.1f} min, nodi espansi A* {a[:,2].mean():.0f} vs senza euristica {a[:,3].mean():.0f}, costo identico nel {a[:,4].mean()*100:.0f}% dei casi")

if __name__ == "__main__":
    main(sys.argv[1])
