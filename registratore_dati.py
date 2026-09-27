"""Registrazione di metriche derivate in CSV e, opzionalmente, Parquet."""

import csv
import os


class RegistratoreDati:
    def __init__(self, cartella, nome_base, intestazione, scrivi_parquet=False):
        self.cartella = cartella
        self.intestazione = intestazione
        self.scrivi_parquet = scrivi_parquet
        self.righe_parquet = []
        os.makedirs(cartella, exist_ok=True)
        self.file_csv = self._nome_compatibile(os.path.join(cartella, f"{nome_base}.csv"))
        self._inizializza_csv()

    def _nome_compatibile(self, percorso):
        if not os.path.isfile(percorso):
            return percorso
        with open(percorso, newline="", encoding="utf-8") as file:
            if next(csv.reader(file), []) == self.intestazione:
                return percorso
        base, estensione = os.path.splitext(percorso)
        for indice in range(2, 100):
            candidato = f"{base}_v{indice}{estensione}"
            if not os.path.isfile(candidato):
                return candidato
        raise RuntimeError("Impossibile trovare un nome di dataset compatibile.")

    def _inizializza_csv(self):
        if not os.path.isfile(self.file_csv):
            with open(self.file_csv, "w", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=self.intestazione).writeheader()

    def salva(self, riga):
        with open(self.file_csv, "a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=self.intestazione).writerow(riga)
        if self.scrivi_parquet:
            self.righe_parquet.append(riga.copy())

    def chiudi(self):
        if not self.scrivi_parquet or not self.righe_parquet:
            return None
        try:
            import pandas as pd
            percorso = os.path.splitext(self.file_csv)[0] + ".parquet"
            pd.DataFrame(self.righe_parquet, columns=self.intestazione).to_parquet(percorso, index=False)
            return percorso
        except (ImportError, ModuleNotFoundError) as errore:
            print("Parquet non creato: installa pandas e pyarrow.", errore)
        except Exception as errore:
            print("Parquet non creato:", errore)
        return None
