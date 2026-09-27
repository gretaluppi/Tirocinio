import json
import os
import time
import uuid
from datetime import datetime

try:
    import msvcrt  # lettura tastiera da terminale, disponibile solo su Windows
except ImportError:
    msvcrt = None

try:
    import cv2
    import mediapipe as mp
except ModuleNotFoundError as errore:
    modulo = errore.name
    pacchetto = "opencv-python" if modulo == "cv2" else modulo
    raise SystemExit(
        f"Dipendenza mancante: {modulo}\n"
        f"Installa le dipendenze con: pip install -r requirements.txt\n"
        f"Oppure installa il pacchetto specifico: pip install {pacchetto}"
    )

from analisi import (
    crea_filtri,
    estrai_blendshapes,
    classifica_emozione,
    analizza_postura,
    analizza_postura_3d,
    inizializza_calibrazione_postura,
    stato_calibrazione_postura,
)
from interfaccia import (
    acquisisci_consenso_privacy,
    acquisisci_codice_persona_da_camera,
    disegna_pannello,
    disegna_debug_blendshapes,
)
from realtime_server import RealtimeServer
from modello_ml import carica_predittore
from registratore_dati import RegistratoreDati
from schema_dati import INTESTAZIONE_CSV, valori_blendshape

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
POSE_MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_full.task")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

def carica_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def inizializza_stato(config):
    raccolta = config.get("raccolta_dati", {})
    return {
        "codice_persona": None,
        "file_csv": None,
        "registratore": None,
        "predittore_ml": None,
        "session_id": str(uuid.uuid4())[:8],
        "filtri": crea_filtri(config),
        # isteresi
        "emozione_candidata": "NEUTRO",
        "tempo_candidata": 0.0,
        "emozione_confermata": "NEUTRO",
        # postura
        "ultimo_stato_posturale": "POSTURA NEUTRA",
        "calibrazione_postura": inizializza_calibrazione_postura(config),
        # salvataggio
        "ultimo_salvataggio": 0.0,
        "intervallo_salvataggio": config["salvataggio"]["intervallo"],
        "etichetta_reale": raccolta.get("etichetta_reale_default", ""),
        "modalita_training": raccolta.get("modalita_training", False),
        "etichette_training": raccolta.get("etichette_training", {}),
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


def pulisci_dati_vecchi(cartella_dati, giorni_conservazione):
    """Controlla i file nella cartella dati e segnala quelli piu' vecchi del
    limite di conservazione impostato in config.json (privacy.conservazione_giorni).
    Di default SEGNALA soltanto: non cancella nulla in automatico, per sicurezza.
    Se vuoi la cancellazione automatica, togli il commento dalla riga os.remove(...)."""
    if not os.path.isdir(cartella_dati):
        return

    limite_secondi = giorni_conservazione * 86400
    adesso = time.time()

    for nome_file in os.listdir(cartella_dati):
        percorso = os.path.join(cartella_dati, nome_file)
        if not os.path.isfile(percorso):
            continue
        eta_secondi = adesso - os.path.getmtime(percorso)
        if eta_secondi > limite_secondi:
            eta_giorni = int(eta_secondi // 86400)
            print(f"[PRIVACY] '{nome_file}' ha {eta_giorni} giorni, oltre il limite di {giorni_conservazione}.")
            # os.remove(percorso)  # scommenta per eliminare automaticamente


def salva_ricevuta_consenso(cartella_dati, session_id, versione_informativa="1.0"):
    """Registra una 'ricevuta' minima del consenso dato (senza nomi), utile
    per dimostrare che il consenso e' stato raccolto prima di registrare dati."""
    os.makedirs(cartella_dati, exist_ok=True)
    percorso = os.path.join(cartella_dati, "consensi.csv")
    file_nuovo = not os.path.exists(percorso)
    with open(percorso, "a", encoding="utf-8", newline="") as f:
        if file_nuovo:
            f.write("session_id,data,versione_informativa,confermato\n")
        data = datetime.now().astimezone().isoformat(timespec="seconds")
        f.write(f"{session_id},{data},{versione_informativa},true\n")


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
        output_facial_transformation_matrixes=True,
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


def salva_dati_csv(stato, punteggio, apertura, occhio_sx, occhio_dx,
                   apertura_spalle, inclinazione_spalle, inclinazione_busto,
                   stato_posturale, valence, arousal, head_yaw, head_pitch,
                   head_roll, head_pose_sorgente, attenzione, calibrazione_stato, emozione,
                   sorgente_emozione, confidenza_modello, bs):
    riga = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "session_id": stato["session_id"],
        "codice_persona": stato["codice_persona"],
        "punteggio_sorriso_0_100": round(punteggio, 2),
        "apertura_bocca": round(apertura, 4),
        "occhio_sx": round(occhio_sx, 4),
        "occhio_dx": round(occhio_dx, 4),
        "apertura_spalle": round(apertura_spalle, 4),
        "inclinazione_spalle": round(inclinazione_spalle, 4),
        "inclinazione_busto": round(inclinazione_busto, 4),
        "stato_posturale": stato_posturale,
        "valence": round(valence, 4), "arousal": round(arousal, 4),
        "head_yaw": round(head_yaw, 4), "head_pitch": round(head_pitch, 4),
        "head_roll": round(head_roll, 4), "head_pose_sorgente": head_pose_sorgente,
        "attenzione_schermo": int(bool(attenzione)),
        "calibrazione_postura": calibrazione_stato,
        "etichetta_reale": stato["etichetta_reale"], "emozione": emozione,
        "sorgente_emozione": sorgente_emozione,
        "confidenza_modello": round(confidenza_modello, 4),
    }
    riga.update(valori_blendshape(bs))
    stato["registratore"].salva(riga)


def crea_payload_realtime(stato, punteggio, apertura, occhio_sx, occhio_dx,
                          apertura_spalle, inclinazione_spalle, inclinazione_busto,
                          stato_posturale, valence, arousal, head_yaw, head_pitch,
                          head_roll, head_pose_sorgente, attenzione,
                          calibrazione_stato, calibrazione_progresso, emozione,
                          sorgente_emozione, confidenza_modello):
    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "session_id": stato["session_id"],
        "codice_persona": stato["codice_persona"],
        "punteggio_sorriso_0_100": round(punteggio, 2),
        "apertura_bocca": round(apertura, 4),
        "occhio_sx": round(occhio_sx, 4),
        "occhio_dx": round(occhio_dx, 4),
        "apertura_spalle": round(apertura_spalle, 4),
        "inclinazione_spalle": round(inclinazione_spalle, 4),
        "inclinazione_busto": round(inclinazione_busto, 4),
        "stato_posturale": stato_posturale,
        "valence": round(valence, 4),
        "arousal": round(arousal, 4),
        "head_yaw": round(head_yaw, 4),
        "head_pitch": round(head_pitch, 4),
        "head_roll": round(head_roll, 4),
        "head_pose_sorgente": head_pose_sorgente,
        "attenzione_schermo": bool(attenzione),
        "calibrazione_postura": calibrazione_stato,
        "calibrazione_progresso": round(calibrazione_progresso, 4),
        "etichetta_reale": stato["etichetta_reale"],
        "emozione": emozione,
        "sorgente_emozione": sorgente_emozione,
        "confidenza_modello": round(confidenza_modello, 4),
    }


# --- LOOP PRINCIPALE ---

def leggi_tasto(mostra_finestra):
    """Restituisce il codice del tasto premuto (stesso formato di cv2.waitKey),
    oppure -1 se nessun tasto e' stato premuto. Con la finestra locale attiva
    serve comunque chiamare cv2.waitKey (aggiorna anche l'immagine mostrata).
    Senza finestra (modalita' headless) legge invece la tastiera dal terminale,
    cosi' i tasti Q/D/numeri continuano a funzionare."""
    if mostra_finestra:
        return cv2.waitKey(1) & 0xFF
    if msvcrt is None or not msvcrt.kbhit():
        return -1
    return ord(msvcrt.getch())


def esegui_rilevamento(cap, face_landmarker, pose_landmarker, stato, config, realtime_server=None, mostra_finestra=True):
    print("Webcam aperta")
    print("Codice persona:", stato["codice_persona"])
    print("Session ID:", stato["session_id"])
    print("Salvataggio dati in:", stato["file_csv"])
    if stato["modalita_training"]:
        print("Modalita training attiva. Tasti etichetta reale:")
        for tasto, etichetta in stato["etichette_training"].items():
            nome = etichetta if etichetta else "NESSUNA"
            print(f"  {tasto}: {nome}")
    if not mostra_finestra:
        print("Finestra locale disattivata (config.json -> interfaccia -> mostra_finestra_locale).")
        print("I risultati sono visibili solo sulla dashboard web. I tasti Q/D/numeri funzionano")
        print("comunque, ma serve dare il focus a questa finestra del terminale.")

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
            valence = 0.0
            arousal = 0.0
            head_yaw = 0.0
            head_pitch = 0.0
            head_roll = 0.0
            head_pose_sorgente = "NON_DISPONIBILE"
            sorgente_emozione = "NESSUN_VOLTO"
            confidenza_modello = 0.0
            bs = {}
            attenzione = False
            pose_info = None
            calibrazione_stato, calibrazione_progresso = stato_calibrazione_postura(
                stato["calibrazione_postura"], t
            )

            pose_world_landmarks = getattr(pose_results, "pose_world_landmarks", None)
            if pose_world_landmarks:
                pose_info = analizza_postura_3d(
                    pose_world_landmarks[0], stato["filtri"], config, t,
                    stato["calibrazione_postura"],
                )
                apertura_spalle, inclinazione_spalle, inclinazione_busto, stato_posturale = pose_info
                stato["ultimo_stato_posturale"] = stato_posturale
                calibrazione_stato, calibrazione_progresso = stato_calibrazione_postura(
                    stato["calibrazione_postura"], t
                )
            elif pose_results.pose_landmarks:
                pose_info = analizza_postura(
                    pose_results.pose_landmarks[0], stato["filtri"], config, t
                )
                apertura_spalle, inclinazione_spalle, inclinazione_busto, stato_posturale = pose_info
                stato["ultimo_stato_posturale"] = stato_posturale

            if results.face_blendshapes:
                bs = estrai_blendshapes(results.face_blendshapes[0])
                ultimo_bs = bs
                matrici_facciali = getattr(results, "facial_transformation_matrixes", None)
                matrice_facciale = matrici_facciali[0] if matrici_facciali else None
                predizione_ml = None
                if stato["predittore_ml"] is not None:
                    predizione_ml = stato["predittore_ml"].predici(bs)
                (
                    emozione, punteggio, apertura,
                    occhio_sx, occhio_dx,
                    apertura_spalle, inclinazione_spalle,
                    inclinazione_busto, stato_posturale,
                    valence, arousal, head_yaw, head_pitch, head_roll,
                    attenzione, head_pose_sorgente,
                    sorgente_emozione, confidenza_modello,
                ) = classifica_emozione(
                    bs, stato["filtri"], stato, config, t, pose_info, matrice_facciale,
                    predizione_ml,
                )

                adesso = time.time()
                if adesso - stato["ultimo_salvataggio"] >= stato["intervallo_salvataggio"]:
                    salva_dati_csv(
                        stato,
                        punteggio, apertura, occhio_sx, occhio_dx,
                        apertura_spalle, inclinazione_spalle, inclinazione_busto,
                        stato_posturale, valence, arousal, head_yaw, head_pitch,
                        head_roll, head_pose_sorgente, attenzione, calibrazione_stato, emozione,
                        sorgente_emozione, confidenza_modello, bs,
                    )
                    stato["ultimo_salvataggio"] = adesso

            if realtime_server:
                realtime_server.broadcast(crea_payload_realtime(
                    stato, punteggio, apertura, occhio_sx, occhio_dx,
                    apertura_spalle, inclinazione_spalle, inclinazione_busto,
                    stato_posturale, valence, arousal, head_yaw, head_pitch,
                    head_roll, head_pose_sorgente, attenzione,
                    calibrazione_stato, calibrazione_progresso, emozione,
                    sorgente_emozione, confidenza_modello,
                ))

            if mostra_finestra:
                disegna_pannello(
                    frame, emozione, punteggio, apertura, stato_posturale, stato,
                    valence, arousal, attenzione, calibrazione_stato, calibrazione_progresso,
                    sorgente_emozione, confidenza_modello,
                )
                if stato["mostra_debug"] and ultimo_bs:
                    disegna_debug_blendshapes(frame, ultimo_bs)
                cv2.imshow("Emotion Dataset Recorder", frame)

            key = leggi_tasto(mostra_finestra)
            if key == ord("q") or key == 27:
                break
            elif key == ord("d"):
                stato["mostra_debug"] = not stato["mostra_debug"]
            elif stato["modalita_training"]:
                carattere = chr(key) if 0 <= key <= 255 else ""
                if carattere in stato["etichette_training"]:
                    stato["etichetta_reale"] = stato["etichette_training"][carattere]
                    etichetta = stato["etichetta_reale"] or "NESSUNA"
                    print("Etichetta reale impostata:", etichetta)
    finally:
        parquet = stato["registratore"].chiudi() if stato["registratore"] else None
        if parquet:
            print("Dataset Parquet salvato in:", parquet)
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

    raccolta = config["raccolta_dati"]
    cartella_dati = os.path.join(BASE_DIR, raccolta.get("directory", "dati"))
    stato["registratore"] = RegistratoreDati(
        cartella_dati, "dataset_emozioni", INTESTAZIONE_CSV,
        raccolta.get("scrivi_parquet", False),
    )
    stato["file_csv"] = stato["registratore"].file_csv

    conservazione_giorni = config.get("privacy", {}).get("conservazione_giorni")
    if conservazione_giorni:
        pulisci_dati_vecchi(cartella_dati, conservazione_giorni)

    mostra_finestra = config.get("interfaccia", {}).get("mostra_finestra_locale", True)

    ml_cfg = config.get("ml", {})
    if ml_cfg.get("abilitato", True):
        percorso_modello = os.path.join(BASE_DIR, ml_cfg.get("file_modello", "modello_emozioni.pkl"))
        try:
            stato["predittore_ml"] = carica_predittore(percorso_modello)
            if stato["predittore_ml"]:
                print("Modello ML caricato:", percorso_modello)
            else:
                print("Modello ML non ancora disponibile: uso classificazione euristica.")
        except (RuntimeError, ValueError) as errore:
            print("Modello ML ignorato:", errore)

    print("\nPremi 'q' o ESC per uscire")
    print("Premi 'd' per mostrare/nascondere i blendshapes\n")
    if stato["modalita_training"]:
        print("Training: usa i tasti numerici per impostare etichetta_reale durante la registrazione.")

    realtime_server = None
    if config.get("realtime", {}).get("abilitato", True):
        realtime_cfg = config["realtime"]
        dashboard_dir = os.path.join(BASE_DIR, realtime_cfg.get("dashboard_dir", "dashboard"))
        realtime_server = RealtimeServer(
            realtime_cfg.get("host", "127.0.0.1"),
            realtime_cfg.get("porta", 8765),
            dashboard_dir,
        )
        try:
            realtime_server.start()
            print("Dashboard realtime:", realtime_server.url())
        except OSError as errore:
            print("Dashboard realtime non avviata:", errore)
            print("Il programma continua senza dashboard. Controlla se la porta e' gia' occupata.")
            realtime_server = None

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        if realtime_server:
            realtime_server.stop()
        print("Webcam non disponibile")
        return

    if config.get("privacy", {}).get("richiedi_consenso", True):
        if not acquisisci_consenso_privacy(cap, conservazione_giorni):
            cap.release()
            cv2.destroyAllWindows()
            if realtime_server:
                realtime_server.stop()
            print("Consenso non confermato")
            return
        salva_ricevuta_consenso(cartella_dati, stato["session_id"])

    stato["codice_persona"] = acquisisci_codice_persona_da_camera(cap)
    if not stato["codice_persona"]:
        cap.release()
        cv2.destroyAllWindows()
        if realtime_server:
            realtime_server.stop()
        print("Inserimento codice annullato")
        return

    if not mostra_finestra:
        cv2.destroyAllWindows()

    try:
        esegui_rilevamento(cap, face_landmarker, pose_landmarker, stato, config, realtime_server, mostra_finestra)
    finally:
        if realtime_server:
            realtime_server.stop()


if __name__ == "__main__":
    main()
