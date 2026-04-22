import cv2
import mediapipe as mp
import time
import csv
import os

# ==========================================
# CLASSE RILEVATORE EMOZIONI
# ==========================================

class RilevatoreEmozioni:

    def __init__(self):

        # Nuova API Mediapipe (Tasks)
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Carica il modello FaceLandmarker
        self.options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="face_landmarker.task"),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1
        )

        # Crea il rilevatore
        self.landmarker = FaceLandmarker.create_from_options(self.options)

        # File CSV
        self.file_csv = "dataset_emozioni.csv"
        self.inizializza_csv()

    # --------------------------------------
    # CREA FILE CSV PER SALVARE I DATI
    # --------------------------------------

    def inizializza_csv(self):

        file_esiste = os.path.isfile(self.file_csv)

        with open(self.file_csv, mode='a', newline='') as file:
            writer = csv.writer(file)

            if not file_esiste:
                writer.writerow([
                    "timestamp",
                    "punteggio_sorriso_0_100",
                    "apertura_bocca",
                    "occhio_sx",
                    "occhio_dx",
                    "emozione"
                ])

    # --------------------------------------
    # SALVA RIGA NEL CSV
    # --------------------------------------

    def salva_dati(self, punteggio, apertura, occhio_sx, occhio_dx, emozione):

        timestamp = time.strftime("%H:%M:%S")

        with open(self.file_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp,
                round(punteggio, 2),
                round(apertura, 4),
                round(occhio_sx, 4),
                round(occhio_dx, 4),
                emozione
            ])

    # --------------------------------------
    # CALCOLO SORRISO (NORMALIZZATO 0-100)
    # --------------------------------------

    def calcola_sorriso(self, landmarks):

        centro_y = landmarks[13].y
        angolo_sx_y = landmarks[61].y
        angolo_dx_y = landmarks[291].y

        diff_sx = centro_y - angolo_sx_y
        diff_dx = centro_y - angolo_dx_y

        punteggio_grezzo = 0

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
        punteggio_normalizzato = max(0, min(100, punteggio_normalizzato))

        return punteggio_normalizzato

    # --------------------------------------
    # ANALISI ESPRESSIONE
    # --------------------------------------

    def analizza(self, landmarks):

        punteggio = self.calcola_sorriso(landmarks)

        apertura = abs(landmarks[14].y - landmarks[13].y)
        occhio_sx = abs(landmarks[145].y - landmarks[159].y)
        occhio_dx = abs(landmarks[374].y - landmarks[386].y)

        if apertura > 0.045:
            emozione = "SORPRESO"

        elif punteggio > 75:
            emozione = "MOLTO FELICE"

        elif punteggio > 55:
            emozione = "FELICE"

        elif punteggio < 40:
            emozione = "ARRABBIATO"

        else:
            emozione = "NEUTRO"

        return emozione, punteggio, apertura, occhio_sx, occhio_dx

    # --------------------------------------
    # AVVIO RILEVATORE
    # --------------------------------------

    def avvia(self):

        print("\n🎭 RILEVATORE EMOZIONI + DATASET")
        print("Premi 'q' per uscire\n")

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print("❌ Webcam non disponibile")
            return

        print("✅ Webcam aperta")
        print("📁 Salvataggio dati in:", self.file_csv)

        try:
            while True:

                ret, frame = cap.read()
                if not ret:
                    break

                # Converti in formato Mediapipe
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )

                # Rilevamento volto
                results = self.landmarker.detect(mp_image)

                if results.face_landmarks:
                    for face in results.face_landmarks:

                        landmarks = face  # lista dei 468 punti

                        emozione, punteggio, apertura, occhio_sx, occhio_dx = \
                            self.analizza(landmarks)

                        self.salva_dati(
                            punteggio,
                            apertura,
                            occhio_sx,
                            occhio_dx,
                            emozione
                        )

                        cv2.putText(frame,
                                    f"EMOZIONE: {emozione}",
                                    (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 255, 0),
                                    3)

                        cv2.putText(frame,
                                    f"Sorriso (0-100): {punteggio:.1f}",
                                    (30, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (255, 255, 255),
                                    2)

                cv2.imshow("Emotion Dataset Recorder", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Programma terminato")


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 60)
    print("🤖 SISTEMA RILEVAMENTO EMOZIONI + DATASET")
    print("=" * 60)

    rilevatore = RilevatoreEmozioni()
    rilevatore.avvia()


if __name__ == "__main__":
    main()
