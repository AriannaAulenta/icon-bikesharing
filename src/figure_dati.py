import sys, re, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "src")
import build_kb as B
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
NOMI = {"dc": "Washington", "bos": "Boston", "jc": "Jersey City"}
COL = {"stazione_esportatrice": "#d95f02", "stazione_importatrice": "#1b9e77", "stazione_equilibrata": "#7570b3", "stazione_ricreativa": "#e7298a"}
ETI = {"stazione_esportatrice": "esportatrice", "stazione_importatrice": "importatrice", "stazione_equilibrata": "equilibrata", "stazione_ricreativa": "ricreativa"}

def leggi_kb(c):
    lab, cls = {}, {}
    for r in open(f"kb/generated/{c}.pl", encoding="utf8"):
        m = re.match(r"prop\((s_\w+), profilo, p(\d+)\)\.", r)
        if m: lab[m.group(1)] = int(m.group(2))
        m = re.match(r"prop\(p(\d+), classe_stazioni, (\w+)\)\.", r)
        if m: cls[int(m.group(1))] = m.group(2)
    return lab, cls

# ---- Fig. profili orari (feriale, giorni 1-20) e mappa
fig, ax = plt.subplots(2, 3, figsize=(7.6, 5.2), gridspec_kw={"height_ratios": [1, 1.15]})
for j, c in enumerate(["dc", "bos", "jc"]):
    t, ana = B.carica(c); lab, cls = leggi_kb(c)
    t = t[(t.g <= 20) & (t.tipo == "feriale")].copy(); t["p"] = t.stazione.map(lambda s: lab[B.atomo(s)])
    m = t.groupby(["p", "h"]).saldo.mean().unstack(0)
    n = pd.Series({p: sum(1 for v in lab.values() if v == p) for p in cls})
    a = ax[0, j]
    for i_, p in enumerate(m.columns):
        a.plot(m.index, m[p], color=COL[cls[p]], lw=1.6, alpha=.9, ls=["-", "--", ":", "-.", (0, (5, 1))][i_ % 5], label=f"{ETI[cls[p]]} (p{p}, n={n[p]})")
    a.axhline(0, color="grey", lw=.6); a.set_title(NOMI[c], fontsize=10); a.set_xlabel("ora del giorno"); a.set_xticks(range(0, 24, 4))
    if j == 0: a.set_ylabel("saldo medio orario\n(arrivi − partenze)")
    a.legend(fontsize=5.5, frameon=False, loc="upper left")
    ana = ana.set_index("stazione"); b = ax[1, j]
    for p in sorted(cls):
        ss = [s for s in ana.index if lab.get(B.atomo(s)) == p]
        b.scatter(ana.loc[ss, "lng"], ana.loc[ss, "lat"], s=3 if c != "jc" else 9, color=COL[cls[p]], alpha=.75, linewidths=0)
    b.set_aspect(1/np.cos(np.radians(ana.lat.mean()))); b.set_xlabel("longitudine"); 
    if j == 0: b.set_ylabel("latitudine")
plt.tight_layout(); plt.savefig("figure/fig_profili_mappe.png", dpi=170); plt.close()

# ---- Fig. scelta di k (gomito + BIC)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
scelto = {"dc": 4, "bos": 4, "jc": 5}
for c, colr in zip(["dc", "bos", "jc"], ["#1f77b4", "#d62728", "#2ca02c"]):
    t, ana = B.carica(c); m, act = B.profili(t[t.g <= 20]); X = m.div(act + 0.5, axis=0).fillna(0.0).values
    ks = list(range(2, 10)); inert = []; bics = []
    for k in ks:
        b, km = B.bic_kmeans(X, k); inert.append(km.inertia_); bics.append(b)
    ax[0].plot(ks, np.array(inert)/inert[0], "-o", ms=3, color=colr, label=NOMI[c])
    ax[1].plot(ks, np.array(bics) - bics[0], "-o", ms=3, color=colr, label=NOMI[c])
    ax[0].axvline(scelto[c], color=colr, ls=":", lw=1)
ax[0].set_xlabel("numero di cluster k"); ax[0].set_ylabel("inerzia relativa a k = 2"); ax[0].set_title("Metodo del gomito (linee tratteggiate: k scelto)", fontsize=9); ax[0].legend(frameon=False, fontsize=8)
ax[1].set_xlabel("numero di cluster k"); ax[1].set_ylabel("BIC relativo a k = 2"); ax[1].set_title("BIC: nessun minimo per k ≤ 9", fontsize=9)
plt.tight_layout(); plt.savefig("figure/fig_scelta_k.png", dpi=170); plt.close(); print("ok")
