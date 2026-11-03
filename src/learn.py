"""Apprendimento supervisionato: previsione del saldo orario (arrivi - partenze) per stazione.
Valutazione: nested CV (esterna 5 fold per giorno, interna 3 fold per giorno), media e dev. standard."""
import sys, time, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED, N_STAZ = 0, 60
FEAT = ["ora_sin","ora_cos","giorno_sett","weekend","lat","lng"]

def carica(c):
    t = pd.read_csv(f"data/proc/{c}_orario.csv", parse_dates=["ora"])
    s = pd.read_csv(f"data/proc/{c}_stazioni.csv", dtype={"stazione":str})
    t["stazione"] = t.stazione.astype(str)
    t = t.merge(s[["stazione","lat","lng"]], on="stazione", how="inner")   # solo stazioni con coordinate (che hanno partenze)
    rng = np.random.RandomState(SEED)
    scelte = rng.choice(sorted(t.stazione.unique()), size=min(N_STAZ, t.stazione.nunique()), replace=False)
    t = t[t.stazione.isin(scelte)].copy()
    h = t.ora.dt.hour
    t["ora_sin"], t["ora_cos"] = np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)
    t["giorno_sett"] = t.ora.dt.dayofweek; t["weekend"] = (t.giorno_sett >= 5).astype(int)
    t["giorno"] = t.ora.dt.dayofyear
    return t

def modelli():
    return {
     "Baseline (media)": (DummyRegressor(strategy="mean"), {}),
     "Baseline (mediana)": (DummyRegressor(strategy="median"), {}),
     "Regressione Ridge": (make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha":[0.1, 10.0]}),
     "Albero di regressione": (DecisionTreeRegressor(random_state=SEED), {"max_depth":[4,8,12], "min_samples_leaf":[20,100]}),
     "Random forest": (RandomForestRegressor(n_estimators=40, random_state=SEED, n_jobs=1), {"max_depth":[8,12], "min_samples_leaf":[50]}),
     "Gradient boosting": (GradientBoostingRegressor(n_estimators=60, random_state=SEED), {"max_depth":[2,3], "learning_rate":[0.1]}),
     "Rete neurale (MLP)": (make_pipeline(StandardScaler(), MLPRegressor(max_iter=40, early_stopping=True, random_state=SEED)),
                            {"mlpregressor__hidden_layer_sizes":[(16,),(32,16)], "mlpregressor__alpha":[1e-3]}),
    }

def main(citta):
    out = []
    for c in citta:
        d = carica(c); X, y, g = d[FEAT].values, d["saldo"].values, d["giorno"].values
        print(f"[{c}] righe={len(d)} stazioni={d.stazione.nunique()} std(saldo)={y.std():.3f}", flush=True)
        ext = GroupKFold(n_splits=5)
        for nome, (m, grid) in modelli().items():
            t0 = time.time(); mae, rmse, r2, par = [], [], [], []
            for tr, te in ext.split(X, y, g):
                est = GridSearchCV(m, grid, cv=GroupKFold(n_splits=3), scoring="neg_mean_squared_error", n_jobs=1) if grid else m
                est.fit(X[tr], y[tr], **({"groups": g[tr]} if grid else {}))
                p = est.predict(X[te])
                mae.append(mean_absolute_error(y[te], p)); rmse.append(np.sqrt(mean_squared_error(y[te], p))); r2.append(r2_score(y[te], p))
                par.append(str(est.best_params_) if grid else "-")
            out.append(dict(citta=c, modello=nome, MAE_m=np.mean(mae), MAE_sd=np.std(mae), RMSE_m=np.mean(rmse), RMSE_sd=np.std(rmse),
                            R2_m=np.mean(r2), R2_sd=np.std(r2), iperparametri=" | ".join(par), secondi=round(time.time()-t0,1)))
            print(f"  {nome:24s} MAE {np.mean(mae):.4f}±{np.std(mae):.4f}  R2 {np.mean(r2):.4f}±{np.std(r2):.4f}  ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(out).to_csv("data/proc/risultati_apprendimento.csv", index=False)

if __name__ == "__main__":
    main(sys.argv[1:] or ["dc","jc","bos"])
