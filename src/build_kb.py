"""Genera i fatti della KB (triple prop/3) per una citta' a partire dai dati.
La KB usa SOLO i giorni 1-20 di settembre; i giorni 21-30 restano per la valutazione."""
import sys, re, numpy as np, pandas as pd
from sklearn.cluster import KMeans
FASCE = {"notte":range(0,6),"mattina":range(6,10),"giorno":range(10,16),"sera":range(16,20),"serata":range(20,24)}
LAT_CELL = 0.01   # ~1,1 km
import os
from config import TRAIN_IDX, SEED, SUF, TEST_A, TEST_B
RAGGIO = float(os.environ.get('RAGGIO', 0.4)); Q_ALTO = float(os.environ.get('Q_ALTO', 0.85)); Q_MEDIO = float(os.environ.get('Q_MEDIO', 0.65))
TRAIN_FINO_GIORNO = 20
FESTIVI = {pd.Timestamp("2026-09-07").date()}   # Labor Day (USA)

def atomo(x): return "s_" + re.sub(r"[^A-Za-z0-9]", "_", str(x)).lower()

def carica(c):
    t = pd.read_csv(f"data/proc/{c}_orario.csv", parse_dates=["ora"])
    s = pd.read_csv(f"data/proc/{c}_stazioni.csv", dtype={"stazione":str})
    t["stazione"] = t.stazione.astype(str)
    t = t[t.stazione.isin(s.stazione)].copy()
    t["h"] = t.ora.dt.hour; t["g"] = t.ora.dt.day
    t["tipo"] = np.where((t.ora.dt.dayofweek >= 5) | t.ora.dt.date.isin(FESTIVI), "festivo", "feriale")
    t["fascia"] = t.h.map({h:f for f,r in FASCE.items() for h in r})
    t["traffico"] = t.partenze + t.arrivi
    return t, s

def profili(t_train):
    m = t_train.groupby(["stazione","tipo","fascia"]).saldo.mean().unstack(["tipo","fascia"])
    act = t_train.groupby("stazione").traffico.mean()
    return m, act

def bic_kmeans(X, k, seed=0):
    km = KMeans(n_clusters=k, init="random", n_init=10, random_state=seed).fit(X)
    n, d = X.shape; sse = max(km.inertia_, 1e-9)
    return n*d*np.log(sse/(n*d)) + k*d*np.log(n), km

def nome_classe(cen, cols):
    c = dict(zip(cols, cen)); m = c[("feriale","mattina")]
    if m <= -SOGLIA_CLASSE: return "stazione_esportatrice"
    if m >=  SOGLIA_CLASSE: return "stazione_importatrice"
    return "stazione_ricreativa" if c[("festivo","giorno")] > c[("feriale","giorno")] else "stazione_equilibrata"
SOGLIA_CLASSE = 0.10
SOGLIA_GOMITO = 0.12   # si ferma quando passare a k+1 riduce l'inerzia di meno del 12%

def genera(c, verbose=True):
    t, ana = carica(c)
    tr = t[t.g.isin(TRAIN_IDX + 1)]
    m, act = profili(tr)
    X = m.div(act + 0.5, axis=0).fillna(0.0)          # profilo normalizzato per l'attivita' della stazione
    cols = list(X.columns)
    # scelta di k: regola del gomito (il BIC non ha un minimo a valori plausibili di k su questi dati, cfr. documentazione)
    fits = {}
    for k_ in range(2, 10):
        b_, km_ = bic_kmeans(X.values, k_, seed=SEED); fits[k_] = (b_, km_)
    bics = {k_: round(float(v[0]), 1) for k_, v in fits.items()}
    k = next((k_ for k_ in range(2, 9) if (fits[k_][1].inertia_ - fits[k_+1][1].inertia_) / fits[k_][1].inertia_ < SOGLIA_GOMITO), 8)
    km = fits[k][1]
    classi = [nome_classe(cen, cols) for cen in km.cluster_centers_]
    absv = np.abs(m.values.ravel()); absv = absv[~np.isnan(absv)]
    alto, medio = np.quantile(absv, Q_ALTO), np.quantile(absv, Q_MEDIO)
    ana = ana.set_index("stazione").loc[m.index]
    L = [f"%% KB generata per '{c}' - giorni di addestramento: {sorted((TRAIN_IDX + 1).tolist())}", ":- multifile prop/3.", ":- discontiguous prop/3.",
         f"prop(parametri, raggio_vicinato_km, {RAGGIO}).", "prop(parametri, raggio_furgone_km, 3.0).", "prop(parametri, min_critiche_zona, 2).",
         f"prop(soglie, alto, {alto:.4f}).", f"prop(soglie, medio, {medio:.4f})."]
    for i, cl in enumerate(classi):
        L += [f"prop(p{i}, type, profilo).", f"prop(p{i}, classe_stazioni, {cl})."]
    zone = set()
    for (sid, row), lab in zip(m.iterrows(), km.labels_):
        a = atomo(sid); r = ana.loc[sid]
        z = f"z_{int(np.floor(r.lat/LAT_CELL))}_{int(np.floor(abs(r.lng)/LAT_CELL))}".replace("-","m")
        zone.add(z)
        nome = str(r.nome).replace("\\","").replace("'","\\'")
        L += [f"prop({a}, nome, '{nome}').", f"prop({a}, lat, {r.lat:.6f}).", f"prop({a}, lng, {r.lng:.6f}).",
              f"prop({a}, zona_di, {z}).", f"prop({a}, profilo, p{lab})."]
        for (tp, fa), v in row.items():
            if not np.isnan(v): L.append(f"prop({a}, saldo_{tp}_{fa}, {v:.4f}).")
    L += [f"prop({z}, type, zona)." for z in sorted(zone)]
    open(f"kb/generated/{c}{SUF}.pl", "w").write("\n".join(L) + "\n")
    info = dict(citta=c, stazioni=len(m), zone=len(zone), k=k, bic=bics, classi={cl: int((np.array(classi)[km.labels_]==cl).sum()) for cl in set(classi)},
                soglia_alta=round(float(alto),3), soglia_media=round(float(medio),3), fatti=len(L))
    if verbose:
        print(info)
        print("centroidi (feriale mattina, feriale sera):", [(round(cen[cols.index(('feriale','mattina'))],3), round(cen[cols.index(('feriale','sera'))],3)) for cen in km.cluster_centers_])
    return info

if __name__ == "__main__":
    for c in (sys.argv[1:] or ["dc","jc","bos"]): genera(c)
