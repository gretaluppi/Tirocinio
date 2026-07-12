import csv
import json
import os
from collections import Counter


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset_emozioni.csv")
MODEL_PATH = os.path.join(BASE_DIR, "modello_emozioni.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "report_training.json")

FEATURES = [
    "punteggio_sorriso_0_100",
    "apertura_bocca",
    "occhio_sx",
    "occhio_dx",
    "apertura_spalle",
    "inclinazione_spalle",
    "inclinazione_busto",
    "valence",
    "arousal",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "attenzione_schermo",
]


def importa_dipendenze_ml():
    try:
        from joblib import dump
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import GroupShuffleSplit
    except ModuleNotFoundError as errore:
        raise SystemExit(
            f"Dipendenza mancante: {errore.name}\n"
            "Installa le dipendenze ML con: pip install scikit-learn joblib"
        )

    return dump, RandomForestClassifier, accuracy_score, classification_report, confusion_matrix, GroupShuffleSplit


def trova_dataset():
    candidati = [
        os.path.join(BASE_DIR, nome)
        for nome in os.listdir(BASE_DIR)
        if nome.startswith("dataset_emozioni") and nome.endswith(".csv")
    ]
    if not candidati:
        raise FileNotFoundError("Nessun dataset_emozioni*.csv trovato.")
    return max(candidati, key=os.path.getmtime)


def leggi_dataset(path):
    righe = []
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("calibrazione_postura") not in ("COMPLETATA", "DISATTIVATA"):
                continue
            etichetta = row.get("etichetta_reale", "").strip()
            if not etichetta:
                continue
            try:
                features = [float(row[nome]) for nome in FEATURES]
            except (TypeError, ValueError, KeyError):
                continue
            righe.append({
                "features": features,
                "etichetta": etichetta,
                "persona": row.get("codice_persona", "SCONOSCIUTA"),
            })
    return righe


def main():
    dump, RandomForestClassifier, accuracy_score, classification_report, confusion_matrix, GroupShuffleSplit = importa_dipendenze_ml()

    dataset_path = trova_dataset()
    righe = leggi_dataset(dataset_path)
    if len(righe) < 30:
        raise SystemExit("Dataset troppo piccolo: servono almeno 30 righe etichettate valide per un primo training.")

    persone = sorted({r["persona"] for r in righe})
    if len(persone) < 2:
        raise SystemExit("Servono almeno 2 persone diverse per separare training e test per soggetto.")

    x = [r["features"] for r in righe]
    y = [r["etichetta"] for r in righe]
    gruppi = [r["persona"] for r in righe]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(x, y, groups=gruppi))

    x_train = [x[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    x_test = [x[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    modello = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
    )
    modello.fit(x_train, y_train)
    predizioni = modello.predict(x_test)

    report = {
        "dataset": dataset_path,
        "righe_totali_usate": len(righe),
        "persone": persone,
        "distribuzione_etichette": dict(Counter(y)),
        "feature": FEATURES,
        "accuracy": accuracy_score(y_test, predizioni),
        "classification_report": classification_report(y_test, predizioni, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predizioni).tolist(),
        "classi": sorted(set(y)),
    }

    dump({"model": modello, "features": FEATURES}, MODEL_PATH)
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("Training completato")
    print("Dataset:", dataset_path)
    print("Modello salvato in:", MODEL_PATH)
    print("Report salvato in:", REPORT_PATH)
    print("Accuracy:", round(report["accuracy"], 3))


if __name__ == "__main__":
    main()
