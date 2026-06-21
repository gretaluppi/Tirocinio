import csv
import json
import os
import time
import uuid

import cv2
import mediapipe as mp

from analisi import (
    crea_filtri,
    estrai_blendshapes,
    classifica_emozione,
    analizza_postura,
)
from interfaccia import (
    acquisisci_codice_persona_da_camera,
    disegna_pannello,
    disegna_debug_blendshapes,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
POSE_MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_full.task")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

INTESTAZIONE_CSV = [
    "timestamp", "session_id", "codice_persona",
    "punteggio_sorriso_0_100", "apertura_bocca",
    "occhio_sx", "occhio_dx",
    "apertura_spalle", "inclinazione_spalle", "inclinazione_busto",
    "stato_posturale", "emozione",
]


def carica_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def inizializza_stato(config):
    return {
        "codice_persona": None,
        "file_csv": None,
        "session_id": str(uuid.uuid4())[:8],
        "filtri": crea_filtri(config),
        # isteresi
        "emozione_candidata": "NEUTRO",
        "tempo_candidata": 0.0,
        "emozione_confermata": "NEUTRO",
        # postura
        "ultimo_stato_posturale": "POSTURA NEUTRA",
        # salvataggio
        "ultimo_salvataggio": 0.0,
        "intervallo_salvataggio": config["salvataggio"]["intervallo"],
        # debug
        "mostra_debug": False,
    }


def verifica_modelli(face_model_path, pose_model_path):
    modelli_mancanti = []
    if not os.path.exists(face_model_path):
        modelli_mancanti.append(face_model_path)
    if not os.path.exists(pose_model_path):
        modelli_mancanti.append(pose_model_path)
    if modelli_mancanti:
        raise FileNotFoundError("Modelli mancanti:\n" + "\n".join(modelli_mancanti))


def crea_face_landmarker(face_model_path):
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=face_model_path),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
    )
    return FaceLandmarker.create_from_options(options)


def crea_pose_landmarker(pose_model_path):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pose_model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


# --- CSV ---

def prepara_file_csv(nome_file):
    if not os.path.isfile(nome_file):
        return nome_file

    with open(nome_file, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        intestazione_corrente = next(reader, [])

    if intestazione_corrente == INTESTAZIONE_CSV:
        return nome_file

    nuovo_file = "dataset_emozioni_con_codice.csv"
    print(f"CSV esistente con struttura diversa. Nuovi dati in: {nuovo_file}")
    return nuovo_file


def inizializza_csv(file_csv):
    if not os.path.isfile(file_csv):
        with open(file_csv, mode="w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(INTESTAZIONE_CSV)


def salva_dati_csv(file_csv, stato, punteggio, apertura, occhio_sx, occhio_dx,
                   apertura_spalle, inclinazione_spalle, inclinazione_busto,
                   stato_posturale, emozione):
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    with open(file_csv, mode="a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow([
            timestamp, stato["session_id"], stato["codice_persona"],
            round(punteggio, 2), round(apertura, 4),
            round(occhio_sx, 4), round(occhio_dx, 4),
            round(apertura_spalle, 4), round(inclinazione_spalle, 4),
            round(inclinazione_busto, 4),
            stato_posturale, emozione,
        ])


# --- LOOP PRINCIPALE ---

def esegui_rilevamento(cap, face_landmarker, pose_landmarker, stato, config):
    print("Webcam aperta")
    print("Codice persona:", stato["codice_persona"])
    print("Session ID:", stato["session_id"])
    print("Salvataggio dati in:", stato["file_csv"])

    ultimo_bs = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(time.time() * 1000)
            t = time.time()
            results = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_results = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            emozione = stato["emozione_confermata"]
            punteggio = 50.0
            apertura = 0.0
            occhio_sx = 0.0
            occhio_dx = 0.0
            apertura_spalle = 0.0
            inclinazione_spalle = 0.0
            inclinazione_busto = 0.0
            stato_posturale = stato["ultimo_stato_posturale"]
            pose_info = None

            if pose_results.pose_landmarks:
                pose_info = analizza_postura(
                    pose_results.pose_landmarks[0], stato["filtri"], config, t
                )
                apertura_spalle, inclinazione_spalle, inclinazione_busto, stato_posturale = pose_info
                stato["ultimo_stato_posturale"] = stato_posturale

            if results.face_blendshapes:
                bs = estrai_blendshapes(results.face_blendshapes[0])
                ultimo_bs = bs
                (
                    emozione, punteggio, apertura,
                    occhio_sx, occhio_dx,
                    apertura_spalle, inclinazione_spalle,
                    inclinazione_busto, stato_posturale,
                ) = classifica_emozione(bs, stato["filtri"], stato, config, t, pose_info)

                adesso = time.time()
                if adesso - stato["ultimo_salvataggio"] >= stato["intervallo_salvataggio"]:
                    salva_dati_csv(
                        stato["file_csv"], stato,
                        punteggio, apertura, occhio_sx, occhio_dx,
                        apertura_spalle, inclinazione_spalle, inclinazione_busto,
                        stato_posturale, emozione,
                    )
                    stato["ultimo_salvataggio"] = adesso

            disegna_pannello(frame, emozione, punteggio, apertura, stato_posturale, stato)

            if stato["mostra_debug"] and ultimo_bs:
                disegna_debug_blendshapes(frame, ultimo_bs)

            cv2.imshow("Emotion Dataset Recorder", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("d"):
                stato["mostra_debug"] = not stato["mostra_debug"]
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Programma terminato")


def main():
    print("=" * 60)
    print("EMOTIONAL MIRRORING v2")
    print("=" * 60)

    config = carica_config()
    stato = inizializza_stato(config)
    verifica_modelli(FACE_MODEL_PATH, POSE_MODEL_PATH)

    face_landmarker = crea_face_landmarker(FACE_MODEL_PATH)
    pose_landmarker = crea_pose_landmarker(POSE_MODEL_PATH)

    stato["file_csv"] = prepara_file_csv("dataset_emozioni.csv")
    inizializza_csv(stato["file_csv"])

    print("\nPremi 'q' o ESC per uscire")
    print("Premi 'd' per mostrare/nascondere i blendshapes\n")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Webcam non disponibile")
        return

    stato["codice_persona"] = acquisisci_codice_persona_da_camera(cap)
    if not stato["codice_persona"]:
        cap.release()
        cv2.destroyAllWindows()
        print("Inserimento codice annullato")
        return

    esegui_rilevamento(cap, face_landmarker, pose_landmarker, stato, config)


if __name__ == "__main__":
    main()
