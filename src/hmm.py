"""HMM stazionario (cap. 9.6.2): P(S0), P(S_t+1|S_t) uguale per ogni t, P(O_t|S_t,t) con dipendenza dalla fascia oraria.
Apprendimento con EM per BN con variabili latenti (10.4.1) + pseudo-conteggi di Laplace.
Stato nascosto: regime di pressione della stazione; osservazione: saldo orario discretizzato in 5 simboli."""
import sys, numpy as np, pandas as pd
FAS_DI_ORA = np.array([0]*6 + [1]*4 + [2]*6 + [3]*4 + [4]*4)      # notte, mattina, giorno, sera, serata
NSIM, NFAS, T = 5, 5, 24
from config import TRAIN_IDX, TEST_IDX, SEED, SUF
TRAIN_FINO, FESTIVI = 20, {7}

def simbolo(saldo):  # <=-2, -1, 0, 1, >=2
    return np.clip(saldo, -2, 2).astype(int) + 2

def carica_seq(c):
    t = pd.read_csv(f"data/proc/{c}_orario.csv", parse_dates=["ora"], low_memory=False)
    s = pd.read_csv(f"data/proc/{c}_stazioni.csv", dtype={"stazione":str}); t["stazione"] = t.stazione.astype(str)
    t = t[t.stazione.isin(s.stazione)].sort_values(["stazione","ora"])
    st = sorted(t.stazione.unique()); S = len(st)
    saldo = t.saldo.values.reshape(S, 30, 24)
    giorni = np.arange(1, 31); weekend = np.array([pd.Timestamp(2026,9,int(g)).dayofweek >= 5 or int(g) in FESTIVI for g in giorni])
    return st, saldo, giorni, weekend

def fb(O, pi, A, B, fas, backward=True):
    """Inferenza esatta sulla catena (eliminazione di variabili in avanti/indietro nel tempo). A: (K,K) stazionaria; B: (fascia,K,simbolo)."""
    N, Tt = O.shape; K = len(pi)
    alpha = np.zeros((N, Tt, K)); c = np.zeros((N, Tt))
    a = pi[None, :] * B[fas[0]][:, O[:, 0]].T; c[:, 0] = a.sum(1); alpha[:, 0] = a / c[:, 0, None]
    for t in range(1, Tt):
        a = (alpha[:, t-1] @ A) * B[fas[t]][:, O[:, t]].T; c[:, t] = a.sum(1); alpha[:, t] = a / c[:, t, None]
    if not backward: return alpha, c, None
    beta = np.ones((N, Tt, K))
    for t in range(Tt-2, -1, -1):
        beta[:, t] = ((B[fas[t+1]][:, O[:, t+1]].T * beta[:, t+1]) @ A.T) / c[:, t+1, None]
    return alpha, c, beta

def em(O, K, fas, iters=25, seed=0, pseudo=1.0):
    rng = np.random.RandomState(seed)
    pi = rng.dirichlet(np.ones(K)); A = rng.dirichlet(np.ones(K), size=K); B = rng.dirichlet(np.ones(NSIM), size=(NFAS, K))
    for _ in range(iters):
        alpha, c, beta = fb(O, pi, A, B, fas)
        gamma = alpha * beta
        pi = gamma[:, 0].sum(0) + pseudo; pi /= pi.sum()
        xi = np.zeros((K, K))
        for t in range(1, O.shape[1]):
            xi += np.einsum('ni,ij,nj,n->ij', alpha[:, t-1], A, B[fas[t]][:, O[:, t]].T * beta[:, t], 1.0 / c[:, t])
        A = xi + pseudo; A /= A.sum(1, keepdims=True)
        cnt = np.zeros((NFAS, K, NSIM))
        for f in range(NFAS):
            tix = np.flatnonzero(fas == f); Of = O[:, tix]; Gf = gamma[:, tix, :]
            for s in range(NSIM): cnt[f, :, s] = Gf[Of == s].sum(0)
        B = cnt + pseudo; B /= B.sum(2, keepdims=True)
    alpha, c, _ = fb(O, pi, A, B, fas, backward=False)
    return dict(pi=pi, A=A, B=B, ll=float(np.log(c).sum()))

def npar(K): return (K-1) + K*(K-1) + NFAS*K*(NSIM-1)

def log_loss_hmm(m, O):   # log-loss di previsione a un passo, t>=1 (per sequenza)
    _, c, _ = fb(O, m["pi"], m["A"], m["B"], FAS_DI_ORA, backward=False)
    return -np.log(c[:, 1:]).mean(1)

def baseline_ora(Otr):    # distribuzione empirica per ora del giorno (Laplace)
    P = np.stack([(np.bincount(Otr[:, h], minlength=NSIM) + 1.0) for h in range(T)]); return P / P.sum(1, keepdims=True)
def log_loss_base(P, O): return -np.log(P[np.arange(T)[None, 1:], O[:, 1:]]).mean(1)

def main(c):
    st, saldo, giorni, weekend = carica_seq(c)
    Otr = simbolo(saldo[:, TRAIN_IDX, :]).reshape(-1, T); Ote = simbolo(saldo[:, TEST_IDX, :])
    print(f"[{c}] sequenze train={len(Otr)} (stazione-giorno), simboli: {np.bincount(Otr.ravel(), minlength=NSIM)/Otr.size}", flush=True)
    res = {}
    for K in (2, 3, 4):
        best = max((em(Otr, K, FAS_DI_ORA, seed=SEED * 10 + s) for s in range(3)), key=lambda m: m["ll"])   # 3 restart casuali
        bic = -2*best["ll"] + npar(K)*np.log(Otr.size); res[K] = (bic, best); print(f"  K={K}: loglik={best['ll']:.0f} parametri={npar(K)} BIC={bic:.0f}", flush=True)
    Kb = min(res, key=lambda k: res[k][0]); m = res[Kb][1]
    # valutazione sui giorni 21-30: log-loss medio per giorno (media su stazioni e ore)
    rows = []
    for d in range(Ote.shape[1]):
        O = Ote[:, d, :]; rows.append((log_loss_hmm(m, O).mean(), log_loss_base(baseline_ora(Otr), O).mean()))
    r = np.array(rows)
    print(f"  K scelto={Kb}  log-loss a un passo, giorni 21-30: HMM {r[:,0].mean():.4f}±{r[:,0].std():.4f} | baseline per ora {r[:,1].mean():.4f}±{r[:,1].std():.4f} | differenza media {np.mean(r[:,1]-r[:,0]):.4f} (positiva = HMM migliore; HMM migliore in {int((r[:,0]<r[:,1]).sum())}/10 giorni)")
    np.savez(f"data/proc/hmm_{c}{SUF}.npz", K=Kb, **{k: m[k] for k in ("pi","A","B")})

if __name__ == "__main__":
    main(sys.argv[1])
