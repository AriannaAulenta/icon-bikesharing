# Riequilibrio intelligente di reti di bike sharing

Progetto di Ingegneria della Conoscenza (Università degli Studi di Bari Aldo Moro).
Sistema basato su conoscenza che, per tre reti di bike sharing (Washington, Boston, Jersey City), rappresenta le stazioni con una KB in Prolog,
apprende a prevedere lo squilibrio orario (modelli supervisionati, k-means, HMM con EM), assegna le stazioni a 3 furgoni (CSP con ricerca locale)
e ne pianifica il percorso (A*). La documentazione completa è in `docs/`.

**Gruppo:** [nome cognome – matricola] · [aggiungere tutti i componenti]

## Dati (non inclusi)
Scaricare i file dei viaggi di **settembre 2026** dalle pagine «System Data» dei gestori e metterli in queste cartelle:

| Rete | Pagina | File atteso |
|---|---|---|
| Washington | capitalbikeshare.com/system-data | `data/dc/202609-capitalbikeshare-tripdata.csv` |
| Jersey City | pagina System Data di Citi Bike (file con «JC» nel nome) | `data/jc/JC-202609-citibike-tripdata.csv` |
| Boston | bluebikes.com/system-data | `data/bos/202609-bluebikes-tripdata.csv` |

I dati sono distribuiti dai gestori secondo le rispettive licenze d'uso e non vanno ridistribuiti.

## Requisiti
Python 3, SWI-Prolog, pyswip, scikit-learn, pandas, numpy, matplotlib. Eseguire i comandi dalla cartella principale del repository.

## Esecuzione
1. `python3 src/prep.py` – conteggi orari per stazione
2. `python3 src/learn.py` e `python3 src/mlp_sgd.py <rete>` (una rete per volta, `dc`, `bos` o `jc`) – apprendimento supervisionato con validazione annidata
3. `./run_all.sh` – KB, HMM globale e per profilo, valutazione complessiva e del CSP (test: giorni 21–30, seme 0)
4. `./run_robustezza.sh` – altre ripartizioni (test 1–10 e 11–20) e semi 1 e 2 (variabili `TEST_A`, `TEST_B`, `SEED`, `TAG` in `src/config.py`)
5. `./run_sensibilita.sh` – sensibilità al raggio di vicinato e alle soglie
6. `python3 src/aggrega.py` e `python3 src/figure_*.py` – tabelle aggregate e figure (in `figure/`)

## Struttura
- `kb/rules.pl` – schema e regole della KB (con le varianti per l'ablazione); `kb/generated/` si ricrea con `src/build_kb.py`
- `src/` – preprocessing, KB, apprendimento, HMM, CSP, A*, valutazione
- `results/` – risultati e log degli esperimenti riportati nella documentazione
- `docs/` – documentazione del progetto (Word e PDF)
