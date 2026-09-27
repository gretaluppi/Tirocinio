"""
Script per la gestione dei dati raccolti (privacy by design).

Uso da terminale (nella cartella del progetto):

    python gestisci_dati.py elenca-sessioni
    python gestisci_dati.py elimina-sessione <session_id>

NOTA: se il nome della cartella dati e' diverso da "dati" nel tuo config.json
(campo raccolta_dati -> directory), cambia CARTELLA_DATI qui sotto.
"""

import argparse
import glob
import os

try:
    import pandas as pd
except ModuleNotFoundError:
    raise SystemExit("Manca pandas. Installa con: py -m pip install pandas pyarrow")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARTELLA_DATI = os.path.join(BASE_DIR, "dati")


def _carica(percorso):
    if percorso.endswith(".csv"):
        return pd.read_csv(percorso)
    return pd.read_parquet(percorso)


def _salva(percorso, df):
    if percorso.endswith(".csv"):
        df.to_csv(percorso, index=False)
    else:
        df.to_parquet(percorso, index=False)


def elenca_sessioni(cartella_dati=CARTELLA_DATI):
    file_trovati = glob.glob(os.path.join(cartella_dati, "*.csv"))
    sessioni = set()
    for percorso in file_trovati:
        try:
            df = pd.read_csv(percorso, usecols=["session_id"])
        except (ValueError, FileNotFoundError):
            continue
        sessioni.update(df["session_id"].dropna().unique())

    if not sessioni:
        print("Nessuna sessione trovata in", cartella_dati)
        return
    print("Sessioni trovate:")
    for s in sorted(sessioni):
        print(" -", s)


def elimina_sessione(session_id, cartella_dati=CARTELLA_DATI):
    file_trovati = glob.glob(os.path.join(cartella_dati, "*.csv")) + \
        glob.glob(os.path.join(cartella_dati, "*.parquet"))

    if not file_trovati:
        print("Nessun file trovato in", cartella_dati)
        return

    almeno_una_modifica = False

    for percorso in file_trovati:
        nome_file = os.path.basename(percorso)
        try:
            df = _carica(percorso)
        except Exception as errore:
            print(f"Impossibile leggere {nome_file}: {errore}")
            continue

        if "session_id" not in df.columns:
            continue

        righe_da_tenere = df[df["session_id"] != session_id]
        righe_eliminate = len(df) - len(righe_da_tenere)
        if righe_eliminate == 0:
            continue

        almeno_una_modifica = True
        if len(righe_da_tenere) == 0:
            os.remove(percorso)
            print(f"File eliminato interamente: {nome_file}")
        else:
            _salva(percorso, righe_da_tenere)
            print(f"Rimosse {righe_eliminate} righe (sessione {session_id}) da {nome_file}")

    percorso_consensi = os.path.join(cartella_dati, "consensi.csv")
    if os.path.exists(percorso_consensi):
        df = pd.read_csv(percorso_consensi)
        prima = len(df)
        df = df[df["session_id"] != session_id]
        if len(df) != prima:
            df.to_csv(percorso_consensi, index=False)
            print("Ricevuta di consenso rimossa per la sessione", session_id)
            almeno_una_modifica = True

    if not almeno_una_modifica:
        print("Nessun dato trovato per la sessione", session_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gestione dati - privacy by design")
    sottocomandi = parser.add_subparsers(dest="comando")

    sottocomandi.add_parser("elenca-sessioni", help="Mostra tutte le session_id presenti")

    p_elimina = sottocomandi.add_parser("elimina-sessione", help="Elimina tutti i dati di una sessione")
    p_elimina.add_argument("session_id")

    args = parser.parse_args()

    if args.comando == "elenca-sessioni":
        elenca_sessioni()
    elif args.comando == "elimina-sessione":
        elimina_sessione(args.session_id)
    else:
        parser.print_help()
