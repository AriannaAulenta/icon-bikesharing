"""Rete feed-forward con SGD semplice (§8.2.1): niente Adam/momentum (§8.3.1), numero di epoche scelto in CV interna (§7.4.3)."""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "src")
from learn import carica, FEAT
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
GRID = {"mlpregressor__hidden_layer_sizes": [(16,), (32, 16)], "mlpregressor__max_iter": [20, 60]}
def modello():
    return make_pipeline(StandardScaler(), MLPRegressor(solver="sgd", momentum=0.0, nesterovs_momentum=False, learning_rate="constant",
                         learning_rate_init=0.05, batch_size=64, alpha=1e-3, early_stopping=False, random_state=0))
def main(c):
    d = carica(c); X, y, g = d[FEAT].values, d["saldo"].values, d["giorno"].values
    mae, rmse, r2, par = [], [], [], []; t0 = time.time()
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        est = GridSearchCV(modello(), GRID, cv=GroupKFold(n_splits=3), scoring="neg_mean_squared_error", n_jobs=1)
        est.fit(X[tr], y[tr], groups=g[tr]); p = est.predict(X[te])
        mae.append(mean_absolute_error(y[te], p)); rmse.append(np.sqrt(mean_squared_error(y[te], p))); r2.append(r2_score(y[te], p)); par.append(str(est.best_params_))
    print(f"[{c}] MLP-SGD RMSE {np.mean(rmse):.3f}±{np.std(rmse):.3f}  MAE {np.mean(mae):.3f}±{np.std(mae):.3f}  R2 {np.mean(r2):.3f}±{np.std(r2):.3f}  ({time.time()-t0:.0f}s)\n   iperparametri scelti: {par}", flush=True)
    return dict(citta=c, modello="Rete neurale (SGD)", MAE_m=np.mean(mae), MAE_sd=np.std(mae), RMSE_m=np.mean(rmse), RMSE_sd=np.std(rmse), R2_m=np.mean(r2), R2_sd=np.std(r2), iperparametri=" | ".join(par), secondi=round(time.time()-t0, 1))
if __name__ == "__main__":
    import collections
    rows = [main(c) for c in sys.argv[1:]]
    for r in rows: print(r["citta"], collections.Counter(r["iperparametri"].split(" | ")).most_common())
    pd.DataFrame(rows).to_csv("data/proc/risultati_mlp_sgd_" + "_".join(sys.argv[1:]) + ".csv", index=False)
