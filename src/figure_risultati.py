import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
NOMI = {"dc": "Washington", "bos": "Boston", "jc": "Jersey City"}

# ---- RMSE nested CV
a = pd.read_csv("data/proc/risultati_apprendimento.csv"); m = pd.read_csv("data/proc/risultati_mlp_sgd.csv")
a = pd.concat([a[a.modello != "Rete neurale (MLP)"], m]); a = a[a.modello != "Baseline (mediana)"]
ordine = ["Baseline (media)", "Regressione Ridge", "Albero di regressione", "Random forest", "Gradient boosting", "Rete neurale (SGD)"]
eti = ["media\n(baseline)", "Ridge", "Albero", "Foresta", "Boosting", "Rete\nneurale"]
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=False)
for j, c in enumerate(["dc", "bos", "jc"]):
    x = a[a.citta == c].set_index("modello").loc[ordine]
    col = ["#bbbbbb"] + ["#9ecae1"] + ["#3182bd", "#3182bd", "#6baed6", "#6baed6"]
    ax[j].bar(range(6), x.RMSE_m, yerr=x.RMSE_sd, color=col, capsize=3, edgecolor="none")
    ax[j].set_xticks(range(6)); ax[j].set_xticklabels(eti, fontsize=6); ax[j].set_title(NOMI[c], fontsize=9)
    ax[j].set_ylim(x.RMSE_m.min() * 0.8, x.RMSE_m.max() * 1.08)
    if j == 0: ax[j].set_ylabel("RMSE (media ± dev. standard,\n5 fold esterni)")
plt.tight_layout(); plt.savefig("figure/fig_rmse.png", dpi=170); plt.close()

# ---- HMM log-loss
dati = {"dc": [0.894, 0.062, 1.020, 0.061, 0.862, 0.057, 0.957, 0.055], "bos": [1.060, 0.173, 1.160, 0.125, 1.031, 0.147, 1.111, 0.097], "jc": [1.185, 0.109, 1.265, 0.075, 1.148, 0.073, 1.192, 0.044]}
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6))
nomi = ["HMM\nglobale", "base\nper ora", "HMM per\nprofilo", "base per\nprof. e ora"]; colori = ["#2c7fb8", "#bdbdbd", "#08519c", "#969696"]
for j, c in enumerate(["dc", "bos", "jc"]):
    v = dati[c]; ax[j].bar(range(4), v[0::2], yerr=v[1::2], color=colori, capsize=3)
    ax[j].set_xticks(range(4)); ax[j].set_xticklabels(nomi, fontsize=6); ax[j].set_title(NOMI[c], fontsize=9); ax[j].set_ylim(min(v[0::2]) * 0.8, max(v[0::2]) * 1.08)
    if j == 0: ax[j].set_ylabel("log-loss a un passo\n(media ± dev. st., 10 giorni)")
plt.tight_layout(); plt.savefig("figure/fig_hmm.png", dpi=170); plt.close()

# ---- valutazione complessiva
ris = {"dc": [(.066, .042), (.332, .147), (.611, .263), (.151, .120), (.473, .210)], "bos": [(.104, .073), (.335, .154), (.583, .243), (.193, .112), (.381, .242)], "jc": [(.291, .108), (.466, .198), (.563, .245), (.271, .149), (.601, .285)]}
pol = ["caso", "persist.", "KB\n(regole)", "HMM\nglob.", "HMM\nprofilo"]; col = ["#bdbdbd", "#fdae6b", "#31a354", "#9ecae1", "#08519c"]
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.8), sharey=True)
for j, c in enumerate(["dc", "bos", "jc"]):
    v = ris[c]; ax[j].bar(range(5), [x[0] for x in v], yerr=[x[1] for x in v], color=col, capsize=3)
    ax[j].set_xticks(range(5)); ax[j].set_xticklabels(pol, fontsize=6); ax[j].set_title(NOMI[c], fontsize=9); ax[j].set_ylim(0, 1)
    if j == 0: ax[j].set_ylabel("deficit servito / oracolo\n(media ± dev. st., 30 scenari)")
plt.tight_layout(); plt.savefig("figure/fig_valutazione.png", dpi=170); plt.close()

# ---- architettura
fig, ax = plt.subplots(figsize=(11, 4.8)); ax.set_xlim(0, 100); ax.set_ylim(-1, 46); ax.axis("off")
def box(x, y, w, h, t, fc="#eaf2fb", ec="#2c7fb8", fs=9.5, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.2, ls=ls))
    ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")
def arr(x1, y1, x2, y2): ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color="#444", lw=1.1))
def seg(xs, ys): ax.plot(xs, ys, color="#444", lw=1.1)
ax.text(1, 43, "OFFLINE  –  20 giorni di addestramento", fontsize=11, fontweight="bold", color="#2c7fb8"); ax.text(1, 24.5, "ONLINE  –  giorno di test non visto", fontsize=11, fontweight="bold", color="#d95f02")
box(1, 31, 15, 9, "Viaggi di\nsettembre 2026\n(3 reti)"); box(21, 31, 15, 9, "Preprocessing:\nsaldo orario\nper stazione")
box(41, 36, 17, 7, "k-means: profili\ndi stazione"); box(41, 26.5, 17, 6, "KB in Prolog:\nclassi e regole", fc="#e5f5e0", ec="#31a354")
box(63, 36, 17, 7, "HMM per profilo\n(EM + BIC)")
box(84, 31, 15, 9, "Modelli supervisionati\n(Sez. 2): confronto\nvalutato a parte", fc="white", ec="#888", fs=8.5, ls="--")
arr(16, 35.5, 21, 35.5); arr(36, 36, 41, 39.5); arr(36, 34.5, 41, 30); arr(49.5, 36, 49.5, 32.9); arr(58, 39.5, 63, 39.5)
box(1, 8, 17, 9, "Flussi osservati\nfino all’istante t0", fc="#fff1e6", ec="#d95f02"); box(24, 8, 18, 9, "Punteggio di intervento\nper stazione\n(regole KB oppure HMM)", fc="#fff1e6", ec="#d95f02")
box(48, 8, 16, 9, "CSP + ricerca locale:\nstazioni → 3 furgoni", fc="#fff1e6", ec="#d95f02"); box(70, 8, 14, 9, "A*: ordine delle\nfermate di ogni\nfurgone", fc="#fff1e6", ec="#d95f02"); box(88, 8, 11, 9, "Piano\ndei furgoni", fc="#fde0dd", ec="#c51b8a", bold=True)
arr(18, 12.5, 24, 12.5); arr(42, 12.5, 48, 12.5); arr(64, 12.5, 70, 12.5); arr(84, 12.5, 88, 12.5)
seg([49.5, 49.5], [26.5, 21.5]); seg([71.5, 71.5], [36, 21.5]); seg([71.5, 33], [21.5, 21.5]); arr(33, 21.5, 33, 17.6)
box(24, -0.6, 40, 4.2, "Valutazione: deficit reale servito / deficit servito dall’oracolo", fc="white", ec="#888", fs=8.5)
arr(33, 8, 33, 3.9)
plt.savefig("figure/fig_architettura.png", dpi=170, bbox_inches="tight"); plt.close()

# ---- gerarchia di classi
fig, ax = plt.subplots(figsize=(7.0, 2.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
def box(x, y, w, h, t, fc="#eaf2fb", ec="#2c7fb8", fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.2)); ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=fs)
def arr(x1, y1, x2, y2): ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color="#444", lw=1.1))
box(30, 33, 12, 5, "luogo"); box(10, 22, 14, 5, "stazione"); box(48, 22, 14, 5, "zona"); 
box(0, 9, 24, 5, "stazione_esportatrice", fc="#fff1e6", ec="#d95f02", fs=8); box(27, 9, 24, 5, "stazione_importatrice", fc="#e5f5e0", ec="#1b9e77", fs=8); box(54, 9, 24, 5, "stazione_equilibrata", fc="#efedf5", ec="#7570b3", fs=8)
box(54, 0.5, 24, 5, "stazione_ricreativa", fc="#fde0dd", ec="#e7298a", fs=8)
arr(33, 33, 20, 27.6); arr(39, 33, 52, 27.6); arr(17, 22, 12, 14.6); arr(17, 22, 39, 14.6); arr(20, 22, 62, 14.6); arr(66, 9, 66, 5.9)
ax.text(99, 2, "freccia = subClassOf", fontsize=7, color="#555", ha="right")
plt.savefig("figure/fig_gerarchia.png", dpi=170, bbox_inches="tight"); plt.close(); print("ok")
