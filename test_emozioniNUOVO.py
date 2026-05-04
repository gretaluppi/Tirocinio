import cv2
import mediapipe as mp
import time
import csv
import os
import math

# ==========================================
# CLASSE RILEVATORE EMOZIONI + POSTURA
# ==========================================

class RilevatoreEmozioniPostura:

    def __init__(self):

        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # ---------------------------
        # FACE LANDMARKER
        # ---------------------------
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

        self.face_landmarker = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path="face_landmarker.task"),
                running_mode=VisionRunningMode.IMAGE,
                num_faces=1
            )
        )

        # ---------------------------
        # POSE LANDMARKER
        # ---------------------------
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

        self.pose_landmarker = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
                running_mode=VisionRunningMode.IMAGE
            )
        )

        # ---------------------------
        # CSV
        # ---------------------------
        self.file_csv = "dataset_emozioni_postura.csv"
        self.inizializza_csv()

    # --------------------------------------
    # CREA FILE CSV
    # --------------------------------------

    def inizializza_csv(self):

        file_esiste = os.path.isfile(self.file_csv)

        with open(self.file_csv, mode='a', newline='') as f:
            w = csv.writer(f)
            if not file_esiste:
                w.writerow([
                    "timestamp",
                    "emozione",
                    "punteggio_sorriso_0_100",
                    "apertura_bocca",
                    "occhio_sx",
                    "occhio_dx",
                    "inclinazione_busto",
                    "inclinazione_testa",
                    "postura_chiusura",
                    "postura_label",
                    "emozione_postura_combinata"
                ])

    # --------------------------------------
    # SALVA RIGA CSV
    # --------------------------------------

    def salva_dati(self, emozione, punteggio, apertura, occhio_sx, occhio_dx,
                   inclinazione_busto, inclinazione_testa, postura_chiusura,
                   postura_label, emozione_postura):

        timestamp = time.strftime("%H:%M:%S")

        with open(self.file_csv, mode='a', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                timestamp,
                emozione,
                round(punteggio, 2),
                round(apertura, 4),
                round(occhio_sx, 4),
                round(occhio_dx, 4),
                round(inclinazione_busto, 4),
                round(inclinazione_testa, 4),
                round(postura_chiusura, 4),
                postura_label,
                emozione_postura
            ])

    # --------------------------------------
    # CALCOLO SORRISO (COME IL TUO)
    # --------------------------------------

    def calcola_sorriso(self, lm):

        centro_y = lm[13].y
        angolo_sx_y = lm[61].y
        angolo_dx_y = lm[291].y

        diff_sx = centro_y - angolo_sx_y
        diff_dx = centro_y - angolo_dx_y

        punteggio_grezzo = 0

        if diff_sx > 0 and diff_dx > 0:
            punteggio_grezzo += 20

        punteggio_grezzo += (diff_sx + diff_dx) * 1000

        larghezza = abs(lm[291].x - lm[61].x)
        if larghezza > 0.15:
            punteggio_grezzo += 15

        apertura = abs(lm[14].y - lm[13].y)
        if apertura < 0.02:
            punteggio_grezzo += 10

        punteggio_normalizzato = 50 + punteggio_grezzo
        punteggio_normalizzato = max(0, min(100, punteggio_normalizzato))

        return punteggio_normalizzato

    # --------------------------------------
    # ANALISI ESPRESSIONE
    # --------------------------------------

    def analizza_emozione(self, lm):

        punteggio = self.calcola_sorriso(lm)

        apertura = abs(lm[14].y - lm[13].y)
        occhio_sx = abs(lm[145].y - lm[159].y)
        occhio_dx = abs(lm[374].y - lm[386].y)

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
    # ANALISI POSTURA
    # --------------------------------------

    def analizza_postura(self, pose_landmarks):

        lm = pose_landmarks  # lista di 33 landmark

        if len(lm) < 25:
            return 0.0, 0.0, 0.0, "POSTURA_NON_RILEVATA"

        # inclinazione busto (spalle)
        sx = lm[11]
        dx = lm[12]
        inclinazione_busto = math.degrees(
            math.atan2(dx.y - sx.y, dx.x - sx.x)
        )

        # inclinazione testa (orecchie)
        orecchio_sx = lm[7]
        orecchio_dx = lm[8]
        inclinazione_testa = math.degrees(
            math.atan2(orecchio_dx.y - orecchio_sx.y,
                       orecchio_dx.x - orecchio_sx.x)
        )

        # postura chiusa (spalla-anca)
        spalla_sx = lm[11]
        anca_sx = lm[23]
        spalla_dx = lm[12]
        anca_dx = lm[24]

        dist_sx = abs(spalla_sx.y - anca_sx.y)
        dist_dx = abs(spalla_dx.y - anca_dx.y)
        postura_chiusura = (dist_sx + dist_dx) / 2

        # classificazione postura
        if postura_chiusura < 0.18:
            postura_label = "CHIUSA"
        elif postura_chiusura > 0.26:
            postura_label = "APERTA"
        else:
            postura_label = "NEUTRA"

        return inclinazione_busto, inclinazione_testa, postura_chiusura, postura_label

    # --------------------------------------
    # COMBINAZIONE EMOZIONE + POSTURA
    # --------------------------------------

    def combina_emozione_postura(self, emozione, postura_label):

        if postura_label == "POSTURA_NON_RILEVATA":
            return emozione + " (postura non rilevata)"

        if emozione in ["FELICE", "MOLTO FELICE"] and postura_label == "APERTA":
            return "FELICE E APERTO"

        if emozione == "ARRABBIATO" and postura_label == "CHIUSA":
            return "ARRABBIATO E CHIUSO"

        if emozione == "NEUTRO" and postura_label == "CHIUSA":
            return "NEUTRO MA CHIUSO"

        if emozione == "NEUTRO" and postura_label == "APERTA":
            return "NEUTRO MA APERTO"

        return emozione + " + POSTURA " + postura_label

    # --------------------------------------
    # DISEGNO LANDMARK VOLTO
    # --------------------------------------

    def disegna_face_landmarks(self, frame, lm):

        h, w, _ = frame.shape
        for p in lm:
            x = int(p.x * w)
            y = int(p.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # --------------------------------------
    # DISEGNO LANDMARK POSTURA
    # --------------------------------------

    def disegna_pose_landmarks(self, frame, lm):

        h, w, _ = frame.shape

        def pt(i):
            return int(lm[i].x * w), int(lm[i].y * h)

        # punti principali
        for i in [11, 12, 23, 24, 7, 8]:
            x, y = pt(i)
            cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)

        # spalle
        cv2.line(frame, pt(11), pt(12), (255, 255, 0), 2)
        # busto
        cv2.line(frame, pt(11), pt(23), (255, 255, 0), 2)
        cv2.line(frame, pt(12), pt(24), (255, 255, 0), 2)
        # testa
        cv2.line(frame, pt(7), pt(8), (255, 255, 0), 2)

    # --------------------------------------
    # AVVIO
    # --------------------------------------

    def avvia(self):

        print("\n🎭 RILEVATORE EMOZIONI + POSTURA")
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

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                face_results = self.face_landmarker.detect(mp_image)
                pose_results = self.pose_landmarker.detect(mp_image)

                emozione = "NON_RILEVATA"
                punteggio = apertura = occhio_sx = occhio_dx = 0.0
                inclinazione_busto = inclinazione_testa = postura_chiusura = 0.0
                postura_label = "POSTURA_NON_RILEVATA"

                # volto
                if face_results.face_landmarks:
                    lm_face = face_results.face_landmarks[0]
                    emozione, punteggio, apertura, occhio_sx, occhio_dx = \
                        self.analizza_emozione(lm_face)
                    self.disegna_face_landmarks(frame, lm_face)

                # postura
                if pose_results.pose_landmarks:
                    lm_pose = pose_results.pose_landmarks[0]
                    inclinazione_busto, inclinazione_testa, postura_chiusura, postura_label = \
                        self.analizza_postura(lm_pose)
                    self.disegna_pose_landmarks(frame, lm_pose)

                emozione_postura = self.combina_emozione_postura(emozione, postura_label)

                # salva solo se almeno il volto è stato rilevato
                if emozione != "NON_RILEVATA":
                    self.salva_dati(
                        emozione,
                        punteggio,
                        apertura,
                        occhio_sx,
                        occhio_dx,
                        inclinazione_busto,
                        inclinazione_testa,
                        postura_chiusura,
                        postura_label,
                        emozione_postura
                    )

                # overlay testo
                cv2.putText(frame, f"EMOZIONE: {emozione}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.putText(frame, f"Sorriso: {punteggio:.1f}", (30, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.putText(frame, f"Postura: {postura_label}", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                cv2.putText(frame, f"Combinata: {emozione_postura}", (30, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                cv2.imshow("Emotion + Posture Recorder", frame)

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
    print("🤖 SISTEMA RILEVAMENTO EMOZIONI + POSTURA")
    print("=" * 60)

    rilevatore = RilevatoreEmozioniPostura()
    rilevatore.avvia()


if __name__ == "__main__":
    main()
