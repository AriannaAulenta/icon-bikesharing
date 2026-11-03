import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "src")
import hmm as H, ridistribuzione as R
from build_kb import atomo
from kb import KB
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
c = "bos"; st, saldo, giorni, weekend = H.carica_seq(c)
ana = pd.read_csv(f"data/proc/{c}_stazioni.csv", dtype={"stazione": str}).set_index("stazione").loc[st]
lat, lng = ana.lat.values, ana.lng.values; depot = np.array([lat.mean(), lng.mean()])
kb = KB(c); atomi = [atomo(s) for s in st]; idx = {a: i for i, a in enumerate(atomi)}
d, t0, tp, fa = 21, 7, "feriale", "mattina"      # 22 settembre 2026 (martedi'), ore 7
pr = np.zeros(len(st))
for r in kb.q(f"priorita(S,{tp},{fa},P)"): pr[idx[r["S"]]] = max(pr[idx[r["S"]]], r["P"])
reale = np.maximum(0.0, -saldo[:, d, t0:t0+3].sum(1))
rng = np.random.RandomState(0); pool = np.argsort(-(pr + 1e-6 * rng.rand(len(pr))))[:40]
D = R.haversine(lat[pool], lng[pool]); x, f = R.risolvi_csp(pr[pool], D, rng)
print("violazioni:", R.violazioni(x, D), "| stazioni servite:", int((x > 0).sum()), "| deficit reale servito:", reale[pool[x > 0]].sum(), "su totale candidati", reale[pool].sum(), "| oracolo (top40 reale):", np.sort(reale)[::-1][:24].sum())
# --- esempio di derivazione per la stazione a priorita' piu' alta
i = pool[np.argmax(pr[pool])]; a = atomi[i]
print("ESEMPIO", a, ana.nome.iloc[i] if hasattr(ana, "nome") else "")
for q in [f"prop({a},nome,N)", f"prop({a},type,C), C \\== stazione", f"saldo({a},{tp},{fa},V)", f"rischio_svuotamento({a},{tp},{fa},L)", f"compensata({a},{tp},{fa})", f"critica({a},{tp},{fa},L)", f"priorita({a},{tp},{fa},P)", f"findall(B-Dd, vicina({a},B,Dd), Ls)"]:
    print("  ", q, "->", [{k: (v if not isinstance(v, list) else v[:3]) for k, v in r.items()} for r in kb.q(q, 3)])
# --- mappa del piano
fig, ax = plt.subplots(1, 2, figsize=(7.8, 3.5), gridspec_kw={"width_ratios": [1.25, 1]})
cols = ["#e41a1c", "#377eb8", "#4daf4a"]; a0 = ax[0]
a0.scatter(lng, lat, s=3, color="#cccccc", linewidths=0, label="tutte le stazioni")
a0.scatter(lng[pool], lat[pool], s=6 + 11 * pr[pool], facecolors="none", edgecolors="#555", linewidths=.6, label="40 candidate (dimensione = priorità KB)")
for v in range(1, R.V + 1):
    sel = pool[x == v]
    if len(sel) == 0: continue
    path, g, esp = R.percorso_astar(depot, np.column_stack([lat[sel], lng[sel]]), True)
    ordine = [sel[j] for j in path]; xs = [depot[1]] + [lng[j] for j in ordine]; ys = [depot[0]] + [lat[j] for j in ordine]
    a0.plot(xs, ys, "-", color=cols[v-1], lw=1.4, alpha=.9); a0.scatter(xs[1:], ys[1:], s=22, color=cols[v-1], zorder=3, label=f"furgone {v}: {len(sel)} fermate, {g:.0f} min")
a0.scatter([depot[1]], [depot[0]], marker="*", s=140, color="k", zorder=4, label="deposito (baricentro)")
a0.set_xlim(np.percentile(lng[pool], 1) - .02, np.percentile(lng[pool], 99) + .02); a0.set_ylim(np.percentile(lat[pool], 1) - .015, np.percentile(lat[pool], 99) + .015)
a0.set_xlabel("longitudine"); a0.set_ylabel("latitudine"); a0.legend(fontsize=5.5, frameon=False, loc="lower left"); a0.set_title("Boston, 22 settembre 2026, ore 7: piano dei furgoni", fontsize=9)
# --- efficacia euristica
b = ax[1]; rng2 = np.random.RandomState(1); ns = list(range(2, 9)); ma, mu = [], []
for n in ns:
    ea, eu = [], []
    for _ in range(12):
        c0 = rng2.randint(len(st)); dd = R.haversine(lat[[c0]], lng[[c0]], lat, lng)[0]; near = np.argsort(dd)[:25]; sel = rng2.choice(near, n, replace=False)
        pts = np.column_stack([lat[sel], lng[sel]]); dep = pts.mean(0) + np.array([0.004, 0.004])
        _, _, e1 = R.percorso_astar(dep, pts, True); _, _, e2 = R.percorso_astar(dep, pts, False); ea.append(e1); eu.append(e2)
    ma.append(np.mean(ea)); mu.append(np.mean(eu))
b.plot(ns, mu, "-o", ms=4, color="#bbbbbb", label="senza euristica (h = 0)"); b.plot(ns, ma, "-o", ms=4, color="#08519c", label="A* con euristica"); b.set_yscale("log")
b.set_xlabel("fermate da visitare"); b.set_ylabel("nodi espansi (media di 12 prove)"); b.legend(frameon=False, fontsize=8); b.set_title("Efficacia dell’euristica (insiemi compatti di stazioni)", fontsize=9)
plt.tight_layout(); plt.savefig("figure/fig_esempio_astar.png", dpi=170); plt.close()
print("A*:", [round(v) for v in ma], "UCS:", [round(v) for v in mu])
