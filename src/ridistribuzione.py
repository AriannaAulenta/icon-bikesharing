"""CSP (ricerca locale stocastica con random restart) per assegnare stazioni ai furgoni + A* per il percorso di ogni furgone."""
import heapq, numpy as np
V, Q, DMAX_KM = 3, 8, 3.0           # furgoni, fermate massime per furgone, distanza massima fra fermate dello stesso furgone
VEL_KMH, DETOUR, SERVIZIO_MIN = 20.0, 1.3, 5.0
PEN = 1e3                           # peso delle violazioni dei vincoli nella funzione obiettivo

def haversine(lat, lng, lat2=None, lng2=None):
    lat2 = lat if lat2 is None else lat2; lng2 = lng if lng2 is None else lng2
    p = np.pi/180; a = np.sin((lat2[None,:]-lat[:,None])*p/2)**2 + np.cos(lat[:,None]*p)*np.cos(lat2[None,:]*p)*np.sin((lng2[None,:]-lng[:,None])*p/2)**2
    return 2*6371*np.arcsin(np.minimum(1, np.sqrt(a)))

# ------------------------------- CSP -----------------------------------------
def violazioni(x, D):
    n = 0
    for v in range(1, V+1):
        idx = np.flatnonzero(x == v)
        if len(idx) > Q: n += len(idx) - Q
        if len(idx) > 1: n += int((D[np.ix_(idx, idx)] > DMAX_KM).sum() // 2)
    return n

def obiettivo(x, s, D): return s[x > 0].sum() - PEN * violazioni(x, D)

def goloso(s, D):
    """assegnazione iniziale: stazioni in ordine di punteggio, al primo furgone con posto e compatibile (distanza)"""
    x = np.zeros(len(s), int)
    for i in np.argsort(-s):
        for v in range(1, V+1):
            idx = np.flatnonzero(x == v)
            if len(idx) < Q and (len(idx) == 0 or (D[i, idx] <= DMAX_KM).all()): x[i] = v; break
    return x

def risolvi_csp(s, D, rng, restarts=2, passi=400, p_random=0.15):
    """variabili x_i in {0..V} (0 = non servita); massimizza il punteggio servito rispettando capacita' e compattezza"""
    n = len(s); best_x, best_f = np.zeros(n, int), -np.inf
    for r in range(restarts):
        x = goloso(s, D) if r == 0 else np.zeros(n, int)      # 1o riavvio da soluzione golosa, gli altri da zero
        f = obiettivo(x, s, D)
        if f > best_f: best_f, best_x = f, x.copy()
        for _ in range(passi):
            i = rng.randint(n)
            if rng.rand() < p_random: cand = [rng.randint(V+1)]            # passo casuale (random walk)
            else: cand = range(V+1)                                        # miglior valore per la variabile scelta
            bv, bf = x[i], -np.inf
            for val in cand:
                old = x[i]; x[i] = val; fv = obiettivo(x, s, D); x[i] = old
                if fv > bf: bv, bf = val, fv
            x[i] = bv; f = bf
            if f > best_f: best_f, best_x = f, x.copy()
    return best_x, best_f

# ------------------------------- A* ------------------------------------------
def percorso_astar(depot, pts, euristica=True):
    """Ricerca su stati (posizione, insieme delle fermate gia' visitate). Costo = tempo di viaggio (min).
    h = tempo in linea d'aria verso la fermata non visitata piu' lontana (ammissibile e consistente)."""
    n = len(pts)
    if n == 0: return [], 0.0, 0
    P = np.vstack([depot, pts]); T = haversine(P[:,0], P[:,1]) * DETOUR / VEL_KMH * 60.0   # minuti
    full = (1 << n) - 1
    def h(cur, mask):
        if not euristica or mask == full: return 0.0
        rest = [j+1 for j in range(n) if not mask >> j & 1]
        return max(T[cur, j] for j in rest) / DETOUR   # in linea d'aria (senza fattore di deviazione): ancora piu' ottimista
    pq = [(h(0, 0), 0.0, 0, 0, ())]; best = {(0, 0): 0.0}; espansi = 0
    while pq:
        f, g, cur, mask, path = heapq.heappop(pq)
        if best.get((cur, mask), np.inf) < g - 1e-12: continue
        if mask == full: return list(path), g, espansi
        espansi += 1
        for j in range(n):
            if mask >> j & 1: continue
            m2 = mask | (1 << j); g2 = g + T[cur, j+1]
            if g2 < best.get((j+1, m2), np.inf):
                best[(j+1, m2)] = g2; heapq.heappush(pq, (g2 + h(j+1, m2), g2, j+1, m2, path + (j,)))
    return [], 0.0, espansi
