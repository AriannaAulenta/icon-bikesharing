import re, numpy as np, pandas as pd, json
POL = ["casuale", "persistenza", "KB (regole)", "HMM globale", "HMM per profilo (KB)", "KB senza compensazione", "KB senza pesi", "solo saldo medio"]
CFG = {"f1": "", "f2": "", "f3": "", "s1": "_s1", "s2": "_s2"}
files = {"f3s0": "", "f1s0": "_f1s0", "f2s0": "_f2s0", "f3s1": "_s1", "f3s2": "_s2"}
out = {}
for c in ["dc", "bos", "jc"]:
    d = {k: pd.read_csv(f"data/proc/valutazione_{c}{v}.csv") for k, v in files.items()}
    sp = ["f1s0", "f2s0", "f3s0"]; se = ["f3s0", "f3s1", "f3s2"]
    r = {}
    for p in POL:
        fm = [d[k][p].mean() for k in sp]; sm = [d[k][p].mean() for k in se]
        pool = pd.concat([d[k][p] for k in sp])
        r[p] = dict(split_mean=np.mean(fm), split_sd=np.std(fm, ddof=1), seed_mean=np.mean(sm), seed_sd=np.std(sm, ddof=1), pool_mean=pool.mean(), pool_sd=pool.std(), n=len(pool))
    P = pd.concat([d[k] for k in sp]); n = len(P)
    pair = {f"{a}>{b}": int((P[a] > P[b]).sum()) for a, b in [("HMM per profilo (KB)", "KB (regole)"), ("HMM per profilo (KB)", "persistenza"), ("HMM per profilo (KB)", "casuale"), ("HMM per profilo (KB)", "HMM globale"),
          ("KB (regole)", "solo saldo medio"), ("KB (regole)", "KB senza compensazione"), ("KB (regole)", "KB senza pesi"), ("solo saldo medio", "KB (regole)"), ("solo saldo medio", "persistenza"), ("solo saldo medio", "HMM per profilo (KB)")]}
    out[c] = dict(res=r, pair=pair, n=n)
    print(f"\n=== {c}  (3 ripartizioni x 30 = {n} scenari)  [media tra le 3 ripartizioni ± sd tra ripartizioni | media su tutti gli scenari ± sd | sd tra 3 semi]")
    for p in POL: x = r[p]; print(f"  {p:26s} {x['split_mean']:.3f} ± {x['split_sd']:.3f} | {x['pool_mean']:.3f} ± {x['pool_sd']:.3f} | seeds: {x['seed_mean']:.3f} ± {x['seed_sd']:.3f}")
    print("  confronti a coppie su", n, "scenari:", pair)
json.dump(out, open("data/proc/aggregato.json", "w"), indent=1)
# ---- log-loss HMM e CSP dai log
def parse(log):
    t = open(log).read(); blocks = re.split(r"#{8} ", t); res = {}
    for b in blocks:
        m = re.match(r"(\w+) (\w+) test=", b)
        if not m: continue
        c, tag = m.group(1), m.group(2)
        g = re.search(r"\[(?:\w+)\] .*?HMM ([\d.]+)±([\d.]+) \| baseline per ora ([\d.]+)±([\d.]+)", b) or re.search(r"K scelto=\d.*?HMM ([\d.]+)±([\d.]+) \| baseline per ora ([\d.]+)±([\d.]+)", b)
        p = re.search(r"HMM per profilo ([\d.]+)±([\d.]+) \| baseline per profilo e ora ([\d.]+)±([\d.]+)", b)
        cs = re.search(r"violazioni dei vincoli nella soluzione: (\d+).*?goloso = ([\d.]+) ± ([\d.]+)", b)
        ks = re.findall(r"profilo p\d+: stazioni=\d+ K=(\d)", b)
        res[(c, tag)] = dict(g=[float(x) for x in g.groups()] if g else None, p=[float(x) for x in p.groups()] if p else None, csp=[float(cs.group(1)), float(cs.group(2)), float(cs.group(3))] if cs else None, K=ks)
    return res
R = parse("data/proc/robust.log"); print("\n--- log-loss HMM (media ± sd sui 10 giorni di test di ogni ripartizione) ---")
for c in ["dc", "bos", "jc"]:
    for tag in ["f1s0", "f2s0", "s1", "s2"]:
        x = R.get((c, tag)); print(c, tag, "globale/baseline:", x["g"], "| profilo/baseline:", x["p"], "| K profili:", x["K"], "| CSP:", x["csp"])
