import sys, time; sys.path.insert(0,'src')
from kb import KB
c = sys.argv[1]
t=time.time(); kb=KB(c); print(f"=== {c} (caricata in {time.time()-t:.1f}s)", flush=True)
print("stazioni (via ereditarieta'):", len(kb.q("prop(S,type,stazione)")), "| esportatrici:", len(kb.q("prop(S,type,stazione_esportatrice)")), flush=True)
for tp,fa in [("feriale","mattina"),("feriale","sera"),("festivo","giorno")]:
    t=time.time()
    sv = len(kb.q(f"rischio_svuotamento(S,{tp},{fa},_)")); cmp_ = len(kb.q(f"compensata(S,{tp},{fa})"))
    cr = len(kb.q(f"critica(S,{tp},{fa},_)")); zc = len(kb.q(f"zona_critica(Z,{tp},{fa},N)")); tr = len(kb.q(f"trasferimento(Da,A,{tp},{fa},D)"))
    print(f"{tp:8s}{fa:8s} svuotamento={sv:4d} compensate={cmp_:4d} critiche(NAF)={cr:4d} zone_critiche={zc:3d} trasferimenti={tr:5d} [{time.time()-t:.1f}s]", flush=True)
top = sorted(kb.q("priorita(S,feriale,mattina,P)"), key=lambda r:-r['P'])[:3]
print("top priorita' feriale mattina:", [(r['S'], round(r['P'],2)) for r in top])
