"""Preprocessing: da viaggi grezzi a conteggi orari per stazione (partenze, arrivi, saldo)."""
import pandas as pd

CITTA = {
    "dc":  "data/dc/202609-capitalbikeshare-tripdata.csv",
    "jc":  "data/jc/JC-202609-citibike-tripdata.csv",
    "bos": "data/bos/202609-bluebikes-tripdata.csv",
}
INIZIO, FINE = "2026-09-01", "2026-10-01"   # solo settembre: scarta i viaggi iniziati a fine agosto

def prepara(chiave):
    df = pd.read_csv(CITTA[chiave], low_memory=False,
                     usecols=["started_at","ended_at","start_station_id","end_station_id",
                              "start_station_name","start_lat","start_lng"])
    df["started_at"] = pd.to_datetime(df["started_at"]); df["ended_at"] = pd.to_datetime(df["ended_at"])
    n0 = len(df)
    df = df[(df.started_at >= INIZIO) & (df.started_at < FINE)]
    n_mese = len(df)
    # stazione = id in formato testo (gli id di DC/JC arrivano come float)
    for c in ("start_station_id","end_station_id"):
        df[c] = df[c].astype("string").str.replace(r"\.0$", "", regex=True)
    n_senza_s = df.start_station_id.isna().sum(); n_senza_e = df.end_station_id.isna().sum()
    part = (df.dropna(subset=["start_station_id"])
              .assign(h=lambda d: d.started_at.dt.floor("h"))
              .groupby(["start_station_id","h"]).size().rename("partenze"))
    arr = (df[(df.ended_at >= INIZIO) & (df.ended_at < FINE)].dropna(subset=["end_station_id"])
              .assign(h=lambda d: d.ended_at.dt.floor("h"))
              .groupby(["end_station_id","h"]).size().rename("arrivi"))
    part.index.names = ["stazione","ora"]; arr.index.names = ["stazione","ora"]
    t = pd.concat([part, arr], axis=1).fillna(0).astype(int).reset_index()
    # griglia completa stazione x ora (le ore senza viaggi sono 0, non dati mancanti)
    ore = pd.date_range(INIZIO, FINE, freq="h", inclusive="left")
    st = t.stazione.unique()
    g = pd.MultiIndex.from_product([st, ore], names=["stazione","ora"]).to_frame(index=False)
    t = g.merge(t, how="left", on=["stazione","ora"]).fillna({"partenze":0,"arrivi":0})
    t[["partenze","arrivi"]] = t[["partenze","arrivi"]].astype(int)
    t["saldo"] = t.arrivi - t.partenze
    # anagrafica stazioni (nome, coordinate medie)
    ana = (df.dropna(subset=["start_station_id"]).groupby("start_station_id")
             .agg(nome=("start_station_name","first"), lat=("start_lat","mean"), lng=("start_lng","mean"))
             .reset_index().rename(columns={"start_station_id":"stazione"}))
    t.to_csv(f"data/proc/{chiave}_orario.csv", index=False); ana.to_csv(f"data/proc/{chiave}_stazioni.csv", index=False)
    return dict(citta=chiave, viaggi_file=n0, viaggi_settembre=n_mese, senza_partenza=int(n_senza_s),
                senza_arrivo=int(n_senza_e), stazioni=len(st), righe_orarie=len(t),
                partenze_tot=int(t.partenze.sum()), arrivi_tot=int(t.arrivi.sum()))

if __name__ == "__main__":
    r = pd.DataFrame([prepara(k) for k in CITTA]); print(r.T.to_string())
