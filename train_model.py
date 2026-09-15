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
        from sklearn.model_selection import train_test_split
    except ModuleNotFoundError as errore:
        raise SystemExit(
            f"Dipendenza mancante: {errore.name}\n"
            "Installa le dipendenze ML con: pip install scikit-learn joblib"
        )

    return (
        dump,
        RandomForestClassifier,
        accuracy_score,
        classification_report,
        confusion_matrix,
        GroupShuffleSplit,
        train_test_split,
    )


def trova_dataset():
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(f"Dataset non trovato: {DATASET_PATH}")
    return DATASET_PATH


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
    (
        dump,
        RandomForestClassifier,
        accuracy_score,
        classification_report,
        confusion_matrix,
        GroupShuffleSplit,
        train_test_split,
    ) = importa_dipendenze_ml()

    dataset_path = trova_dataset()
    righe = leggi_dataset(dataset_path)
    if len(righe) == 0:
        raise SystemExit(
            "Nessuna riga etichettata valida trovata. Imposta etichetta_reale durante la registrazione e riprova."
        )

    persone = sorted({r["persona"] for r in righe})
    avvisi = []
    if len(righe) < 10:
        avvisi.append(
            "Dataset molto ridotto: il modello viene addestrato comunque, ma il risultato serve solo a verificare la pipeline."
        )
    if len(righe) < 30:
        avvisi.append(
            "Dataset ridotto: il training va interpretato come prova tecnica, non come validazione robusta."
        )
    if len(persone) < 5:
        avvisi.append(
            "Numero di soggetti limitato: nella relazione descrivere il risultato come studio pilota/esplorativo."
        )

    x = [r["features"] for r in righe]
    y = [r["etichetta"] for r in righe]
    gruppi = [r["persona"] for r in righe]

    if len(righe) < 4 or len(set(y)) < 2:
        x_train = x
        y_train = y
        x_test = x
        y_test = y
        split_usato = "training_senza_test_indipendente"
        avvisi.append(
            "Dati insufficienti per creare un test separato: le metriche sono calcolate sugli stessi dati usati per addestrare."
        )
    elif len(persone) >= 2:
        try:
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
            train_idx, test_idx = next(splitter.split(x, y, groups=gruppi))
            x_train = [x[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            x_test = [x[i] for i in test_idx]
            y_test = [y[i] for i in test_idx]
            split_usato = "group_split_per_persona"
        except ValueError:
            x_train = x
            y_train = y
            x_test = x
            y_test = y
            split_usato = "training_senza_test_indipendente"
            avvisi.append(
                "Divisione per persona non possibile con questi dati: metriche calcolate sui dati di training."
            )
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y if min(Counter(y).values()) >= 2 else None,
        )
        split_usato = "split_random_singolo_soggetto"
        avvisi.append(
            "Un solo soggetto disponibile: il test misura solo coerenza interna, non generalizzazione su persone nuove."
        )

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
        "split_usato": split_usato,
        "avvisi": avvisi,
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
    for avviso in avvisi:
        print("Avviso:", avviso)


if __name__ == "__main__":
    main()
