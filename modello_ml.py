"""Caricamento e uso sicuro del classificatore addestrato localmente."""

import os

from schema_dati import MODEL_FEATURES, valori_blendshape


class PredittoreEmozioni:
    def __init__(self, artefatto):
        self.model = artefatto["model"]
        self.features = artefatto["features"]
        if self.features != MODEL_FEATURES:
            raise ValueError(
                "Il modello non usa lo schema di 52 blendshape previsto. "
                "Esegui di nuovo train_model.py con il dataset aggiornato."
            )

    def predici(self, bs):
        valori = valori_blendshape(bs)
        x = [[valori[nome] for nome in self.features]]
        etichetta = str(self.model.predict(x)[0])
        confidenza = 1.0
        if hasattr(self.model, "predict_proba"):
            confidenza = float(max(self.model.predict_proba(x)[0]))
        return etichetta, confidenza


def carica_predittore(path):
    """Restituisce None se il training non e' ancora stato eseguito."""
    if not os.path.isfile(path):
        return None
    try:
        from joblib import load
    except ModuleNotFoundError as errore:
        raise RuntimeError("Per usare il modello ML installa joblib e scikit-learn.") from errore

    return PredittoreEmozioni(load(path))
