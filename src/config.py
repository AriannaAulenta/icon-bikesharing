"""Configurazione di una esecuzione: giorni di test, seme, etichetta dei file. Valori di default = ripartizione originale (test: giorni 21-30, seme 0)."""
import os, numpy as np
TEST_A = int(os.environ.get("TEST_A", 21)); TEST_B = int(os.environ.get("TEST_B", 30))
SEED = int(os.environ.get("SEED", 0)); TAG = os.environ.get("TAG", ""); SUF = f"_{TAG}" if TAG else ""
TEST_IDX = np.arange(TEST_A - 1, TEST_B)                                   # indici 0-based dei giorni di test
TRAIN_IDX = np.array([d for d in range(30) if d not in set(TEST_IDX.tolist())])
