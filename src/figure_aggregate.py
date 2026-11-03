import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
NOMI = {"dc": "Washington", "bos": "Boston", "jc": "Jersey City"}; COL = {"dc": "#1f77b4", "bos": "#d62728", "jc": "#2ca02c"}
agg = json.load(open("data/proc/aggregato.json"))
# ---- valutazione complessiva: 8 politiche, 3 reti
pol = ["casuale", "persistenza", "HMM globale", "HMM per profilo (KB)", "KB (regole)", "KB senza compensazione", "KB senza pesi", "solo saldo medio"]
eti = ["caso", "persistenza", "HMM\nglobale", "HMM per\nprofilo", "KB\n(regole)", "KB senza\ncompensaz.", "KB senza\npesi", "solo saldo\nmedio"]
fig, ax = plt.subplots(figsize=(7.4, 3.2)); w = 0.27
for i, c in enumerate(["dc", "bos", "jc"]):
    m = [agg[c]["res"][p]["split_mean"] for p in pol]; s = [agg[c]["res"][p]["split_sd"] for p in pol]
    ax.bar(np.arange(8) + (i - 1) * w, m, w, yerr=s, color=COL[c], capsize=1.5, label=NOMI[c], error_kw={"lw": .8})
ax.set_xticks(range(8)); ax.set_xticklabels(eti, fontsize=6.3); ax.set_ylabel("deficit servito / oracolo\n(media ± dev. st. tra 3 ripartizioni)"); ax.set_ylim(0, .8)
ax.axvspan(3.5, 7.5, color="#e5f5e0", alpha=.45, lw=0); ax.text(5.5, .76, "politiche basate sulla KB e sul profilo storico", ha="center", fontsize=6.5, color="#2c6e49")
ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper left"); plt.tight_layout(); plt.savefig("figure/fig_valutazione.png", dpi=170); plt.close()
# ---- HMM: log-loss medio tra le 3 ripartizioni
H = {"dc": [(.936, .037), (1.065, .039), (.905, .038), (1.001, .040)], "bos": [(1.142, .071), (1.225, .057), (1.111, .070), (1.180, .061)], "jc": [(1.237, .046), (1.316, .046), (1.198, .044), (1.249, .053)]}
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6)); nomi = ["HMM\nglobale", "base\nper ora", "HMM per\nprofilo", "base per\nprof. e ora"]; colori = ["#2c7fb8", "#bdbdbd", "#08519c", "#969696"]
for j, c in enumerate(["dc", "bos", "jc"]):
    v = H[c]; ax[j].bar(range(4), [x[0] for x in v], yerr=[x[1] for x in v], color=colori, capsize=3)
    ax[j].set_xticks(range(4)); ax[j].set_xticklabels(nomi, fontsize=6); ax[j].set_title(NOMI[c], fontsize=9); ax[j].set_ylim(min(x[0] for x in v) * .8, max(x[0] for x in v) * 1.06)
    if j == 0: ax[j].set_ylabel("log-loss a un passo\n(media ± dev. st. tra 3 ripartizioni)")
plt.tight_layout(); plt.savefig("figure/fig_hmm.png", dpi=170); plt.close()
# ---- sensibilita' al raggio di vicinato
rag = [0.1, 0.2, 0.4, 0.8]
KB = {"dc": [.637, .641, .636, .464], "bos": [.624, .624, .580, .339], "jc": [.673, .652, .563, .411]}; SS = {"dc": .672, "bos": .615, "jc": .680}
fig, ax = plt.subplots(figsize=(4.6, 2.8))
for c in ["dc", "bos", "jc"]:
    ax.plot(range(4), KB[c], "-o", ms=4, color=COL[c], label=f"{NOMI[c]}: KB (regole)"); ax.axhline(SS[c], color=COL[c], ls=":", lw=1)
ax.set_xticks(range(4)); ax.set_xticklabels([f"{r} km" for r in rag]); ax.set_xlabel("raggio di vicinato della regola di compensazione"); ax.set_ylabel("deficit servito / oracolo"); ax.set_ylim(.3, .72)
ax.text(3.0, .688, "linee puntinate: solo saldo medio", fontsize=6.5, ha="right", color="#444"); ax.legend(frameon=False, fontsize=6.5, loc="lower left")
plt.tight_layout(); plt.savefig("figure/fig_sensibilita.png", dpi=170); plt.close(); print("ok")
