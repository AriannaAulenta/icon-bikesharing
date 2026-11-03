"""Interfaccia Python -> KB Prolog (pyswip)."""
from pyswip import Prolog
class KB:
    def __init__(self, citta, suf=""):
        self.p = Prolog(); self.citta = citta
        self.p.consult("kb/rules.pl"); self.p.consult(f"kb/generated/{citta}{suf}.pl")
    def q(self, s, maxn=None):
        out = []
        for r in self.p.query(s):
            out.append(r)
            if maxn and len(out) >= maxn: break
        return out
