import csv
import os
import time
from collections import deque
import cv2
import mediapipe as mp

FACE_MODEL_PATH = r"C:\Users\Greta\Desktop\Tirocinio\face_landmarker.task"
POSE_MODEL_PATH = r"C:\Users\Greta\Desktop\Tirocinio\pose_landmarker_full.task"

def inizializza_stato():
    return {
        "codice_persona": None,
        "file_csv": None,
        "storia_emozioni": deque(maxlen=10),
        "punteggio_smussato": None,
        "apertura_smussata": None,
        "occhio_sx_smussato": None,
        "occhio_dx_smussato": None,
        "apertura_spalle_smussata": None,
        "inclinazione_spalle_smussata": None,
        "inclinazione_busto_smussata": None,
        "ultima_emozione_stabile": "NEUTRO",
        "ultimo_stato_posturale": "POSTURA NEUTRA",
        "ultimo_salvataggio": 0.0,
        "intervallo_salvataggio": 0.5,
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


def acquisisci_codice_persona_da_camera(cap):
    codice = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (35, 35), (605, 215), (15, 18, 30), -1)
        cv2.rectangle(overlay, (35, 35), (605, 215), (0, 215, 255), 2)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(
            frame,
            "INSERISCI CODICE PERSONA",
            (60, 80),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Digita il codice e premi INVIO per confermare",
            (60, 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "BACKSPACE cancella  |  ESC esce",
            (60, 146),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        box_color = (0, 215, 255) if codice else (120, 120, 120)
        cv2.rectangle(frame, (60, 165), (340, 200), box_color, 2)
        cv2.putText(
            frame,
            codice if codice else "_",
            (72, 190),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Emotion Dataset Recorder", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return None
        if key in (13, 10):
            if codice.strip():
                return codice.strip()
        elif key == 8:
            codice = codice[:-1]
        elif 32 <= key <= 126 and len(codice) < 20:
            codice += chr(key)


def prepara_file_csv(nome_file):
    intestazione_attesa = [
        "timestamp",
        "codice_persona",
        "punteggio_sorriso_0_100",
        "apertura_bocca",
        "occhio_sx",
        "occhio_dx",
        "apertura_spalle",
        "inclinazione_spalle",
        "inclinazione_busto",
        "stato_posturale",
        "emozione",
    ]

    if not os.path.isfile(nome_file):
        return nome_file

    with open(nome_file, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        intestazione_corrente = next(reader, [])

    if intestazione_corrente == intestazione_attesa:
        return nome_file

    nuovo_file = "dataset_emozioni_con_codice.csv"
    print(
        f"CSV esistente con struttura diversa. I nuovi dati verranno salvati in: {nuovo_file}"
    )
    return nuovo_file


def inizializza_csv(file_csv):
    file_esiste = os.path.isfile(file_csv)

    with open(file_csv, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_esiste:
            writer.writerow(
                [
                    "timestamp",
                    "codice_persona",
                    "punteggio_sorriso_0_100",
                    "apertura_bocca",
                    "occhio_sx",
                    "occhio_dx",
                    "apertura_spalle",
                    "inclinazione_spalle",
                    "inclinazione_busto",
                    "stato_posturale",
                    "emozione",
                ]
            )


def salva_dati_csv(
    file_csv,
    codice_persona,
    punteggio,
    apertura,
    occhio_sx,
    occhio_dx,
    apertura_spalle,
    inclinazione_spalle,
    inclinazione_busto,
    stato_posturale,
    emozione,
):
    timestamp = time.strftime("%H:%M:%S")

    with open(file_csv, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                timestamp,
                codice_persona,
                round(punteggio, 2),
                round(apertura, 4),
                round(occhio_sx, 4),
                round(occhio_dx, 4),
                round(apertura_spalle, 4),
                round(inclinazione_spalle, 4),
                round(inclinazione_busto, 4),
                stato_posturale,
                emozione,
            ]
        )


def smussa_valore(valore_corrente, valore_precedente, alpha=0.2):
    if valore_precedente is None:
        return valore_corrente
    return (alpha * valore_corrente) + ((1 - alpha) * valore_precedente)


def calcola_sorriso(landmarks):
    centro_y = landmarks[13].y
    angolo_sx_y = landmarks[61].y
    angolo_dx_y = landmarks[291].y

    diff_sx = centro_y - angolo_sx_y
    diff_dx = centro_y - angolo_dx_y

    punteggio_grezzo = 0.0
    if diff_sx > 0 and diff_dx > 0:
        punteggio_grezzo += 20

    punteggio_grezzo += (diff_sx + diff_dx) * 1000

    larghezza = abs(landmarks[291].x - landmarks[61].x)
    if larghezza > 0.15:
        punteggio_grezzo += 15

    apertura = abs(landmarks[14].y - landmarks[13].y)
    if apertura < 0.02:
        punteggio_grezzo += 10

    punteggio_normalizzato = 50 + punteggio_grezzo
    return max(0, min(100, punteggio_normalizzato))


def stabilizza_metriche(landmarks, stato):
    punteggio = calcola_sorriso(landmarks)
    apertura = abs(landmarks[14].y - landmarks[13].y)
    occhio_sx = abs(landmarks[145].y - landmarks[159].y)
    occhio_dx = abs(landmarks[374].y - landmarks[386].y)

    stato["punteggio_smussato"] = smussa_valore(punteggio, stato["punteggio_smussato"])
    stato["apertura_smussata"] = smussa_valore(apertura, stato["apertura_smussata"])
    stato["occhio_sx_smussato"] = smussa_valore(occhio_sx, stato["occhio_sx_smussato"])
    stato["occhio_dx_smussato"] = smussa_valore(occhio_dx, stato["occhio_dx_smussato"])

    return (
        stato["punteggio_smussato"],
        stato["apertura_smussata"],
        stato["occhio_sx_smussato"],
        stato["occhio_dx_smussato"],
    )


def analizza_postura(pose_landmarks, stato):
    spalla_sx = pose_landmarks[11]
    spalla_dx = pose_landmarks[12]
    anca_sx = pose_landmarks[23]
    anca_dx = pose_landmarks[24]
    naso = pose_landmarks[0]

    centro_spalle_x = (spalla_sx.x + spalla_dx.x) / 2
    centro_spalle_y = (spalla_sx.y + spalla_dx.y) / 2
    centro_anche_x = (anca_sx.x + anca_dx.x) / 2

    apertura_spalle = abs(spalla_dx.x - spalla_sx.x)
    inclinazione_spalle = abs(spalla_dx.y - spalla_sx.y)
    inclinazione_busto = abs(centro_spalle_x - centro_anche_x)
    testa_avanti = max(0.0, naso.y - centro_spalle_y)

    stato["apertura_spalle_smussata"] = smussa_valore(
        apertura_spalle, stato["apertura_spalle_smussata"]
    )
    stato["inclinazione_spalle_smussata"] = smussa_valore(
        inclinazione_spalle, stato["inclinazione_spalle_smussata"]
    )
    stato["inclinazione_busto_smussata"] = smussa_valore(
        inclinazione_busto, stato["inclinazione_busto_smussata"]
    )

    stato_posturale = "POSTURA NEUTRA"
    if (
        stato["apertura_spalle_smussata"] < 0.18
        or stato["inclinazione_busto_smussata"] > 0.035
        or testa_avanti > 0.12
    ):
        stato_posturale = "POSTURA CHIUSA"
    elif (
        stato["apertura_spalle_smussata"] > 0.24
        and stato["inclinazione_spalle_smussata"] < 0.025
        and stato["inclinazione_busto_smussata"] < 0.03
    ):
        stato_posturale = "POSTURA APERTA"

    stato["ultimo_stato_posturale"] = stato_posturale

    return (
        stato["apertura_spalle_smussata"],
        stato["inclinazione_spalle_smussata"],
        stato["inclinazione_busto_smussata"],
        stato_posturale,
    )


def analizza_emozione(landmarks, stato, pose_info=None):
    punteggio, apertura, occhio_sx, occhio_dx = stabilizza_metriche(landmarks, stato)

    if apertura > 0.05:
        emozione = "SORPRESO"
    elif punteggio > 78:
        emozione = "MOLTO FELICE"
    elif punteggio > 58:
        emozione = "FELICE"
    elif punteggio < 38:
        emozione = "ARRABBIATO"
    else:
        emozione = "NEUTRO"

    apertura_spalle = 0.0
    inclinazione_spalle = 0.0
    inclinazione_busto = 0.0
    stato_posturale = stato["ultimo_stato_posturale"]

    if pose_info is not None:
        (
            apertura_spalle,
            inclinazione_spalle,
            inclinazione_busto,
            stato_posturale,
        ) = pose_info

        if stato_posturale == "POSTURA CHIUSA" and emozione in ("NEUTRO", "FELICE"):
            emozione = "TESO"
        elif stato_posturale == "POSTURA APERTA" and emozione == "NEUTRO":
            emozione = "SERENO"

    stato["storia_emozioni"].append(emozione)
    emozione_stabile = max(
        set(stato["storia_emozioni"]), key=stato["storia_emozioni"].count
    )
    stato["ultima_emozione_stabile"] = emozione_stabile

    return (
        emozione_stabile,
        punteggio,
        apertura,
        occhio_sx,
        occhio_dx,
        apertura_spalle,
        inclinazione_spalle,
        inclinazione_busto,
        stato_posturale,
    )


def colore_emozione(emozione):
    colori = {
        "MOLTO FELICE": (0, 215, 255),
        "FELICE": (80, 200, 120),
        "NEUTRO": (220, 220, 220),
        "ARRABBIATO": (70, 70, 255),
        "SORPRESO": (255, 170, 70),
        "TESO": (120, 180, 255),
        "SERENO": (120, 220, 180),
    }
    return colori.get(emozione, (255, 255, 255))


def disegna_pannello(frame, emozione, punteggio, apertura, stato_posturale, stato):
    overlay = frame.copy()
    colore = colore_emozione(emozione)
    h, w, _ = frame.shape

    cv2.rectangle(overlay, (18, 18), (320, 148), (15, 18, 30), -1)
    cv2.rectangle(overlay, (18, 18), (320, 148), colore, 2)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(
        frame,
        "EMOTIONAL MIRRORING",
        (30, 42),
        cv2.FONT_HERSHEY_DUPLEX,
        0.52,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        emozione,
        (30, 73),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        colore,
        2,
        cv2.LINE_AA,
    )

    barra_x, barra_y = 30, 86
    barra_w, barra_h = 180, 12
    riempimento = int((max(0, min(100, punteggio)) / 100) * barra_w)
    cv2.rectangle(frame, (barra_x, barra_y), (barra_x + barra_w, barra_y + barra_h), (90, 90, 90), 1)
    cv2.rectangle(frame, (barra_x, barra_y), (barra_x + riempimento, barra_y + barra_h), colore, -1)

    cv2.putText(
        frame,
        f"Sorriso {punteggio:04.1f}",
        (30, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Bocca {apertura:.3f}",
        (145, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        stato_posturale,
        (30, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (215, 215, 215),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"ID {stato['codice_persona']}",
        (215, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Premi Q o ESC per uscire",
        (w - 260, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )


def esegui_rilevamento(cap, face_landmarker, pose_landmarker, stato):
    print("Webcam aperta")
    print("Codice persona:", stato["codice_persona"])
    print("Salvataggio dati in:", stato["file_csv"])

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(time.time() * 1000)
            results = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_results = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            emozione = stato["ultima_emozione_stabile"]
            punteggio = stato["punteggio_smussato"] if stato["punteggio_smussato"] is not None else 50.0
            apertura = stato["apertura_smussata"] if stato["apertura_smussata"] is not None else 0.0
            occhio_sx = stato["occhio_sx_smussato"] if stato["occhio_sx_smussato"] is not None else 0.0
            occhio_dx = stato["occhio_dx_smussato"] if stato["occhio_dx_smussato"] is not None else 0.0
            apertura_spalle = stato["apertura_spalle_smussata"] if stato["apertura_spalle_smussata"] is not None else 0.0
            inclinazione_spalle = stato["inclinazione_spalle_smussata"] if stato["inclinazione_spalle_smussata"] is not None else 0.0
            inclinazione_busto = stato["inclinazione_busto_smussata"] if stato["inclinazione_busto_smussata"] is not None else 0.0
            stato_posturale = stato["ultimo_stato_posturale"]
            pose_info = None

            if pose_results.pose_landmarks:
                pose_info = analizza_postura(pose_results.pose_landmarks[0], stato)
                (
                    apertura_spalle,
                    inclinazione_spalle,
                    inclinazione_busto,
                    stato_posturale,
                ) = pose_info

            if results.face_landmarks:
                landmarks = results.face_landmarks[0]
                (
                    emozione,
                    punteggio,
                    apertura,
                    occhio_sx,
                    occhio_dx,
                    apertura_spalle,
                    inclinazione_spalle,
                    inclinazione_busto,
                    stato_posturale,
                ) = analizza_emozione(landmarks, stato, pose_info)

                adesso = time.time()
                if adesso - stato["ultimo_salvataggio"] >= stato["intervallo_salvataggio"]:
                    salva_dati_csv(
                        stato["file_csv"],
                        stato["codice_persona"],
                        punteggio,
                        apertura,
                        occhio_sx,
                        occhio_dx,
                        apertura_spalle,
                        inclinazione_spalle,
                        inclinazione_busto,
                        stato_posturale,
                        emozione,
                    )
                    stato["ultimo_salvataggio"] = adesso

            disegna_pannello(frame, emozione, punteggio, apertura, stato_posturale, stato)
            cv2.imshow("Emotion Dataset Recorder", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Programma terminato")


def main():
    print("=" * 60)
    print("SISTEMA RILEVAMENTO EMOZIONI + DATASET")
    print("=" * 60)

    stato = inizializza_stato()
    verifica_modelli(FACE_MODEL_PATH, POSE_MODEL_PATH)

    face_landmarker = crea_face_landmarker(FACE_MODEL_PATH)
    pose_landmarker = crea_pose_landmarker(POSE_MODEL_PATH)

    stato["file_csv"] = prepara_file_csv("dataset_emozioni.csv")
    inizializza_csv(stato["file_csv"])

    print("\nRILEVATORE EMOZIONI + DATASET")
    print("Premi 'q' o ESC per uscire\n")

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

    esegui_rilevamento(cap, face_landmarker, pose_landmarker, stato)


if __name__ == "__main__":
    main()
