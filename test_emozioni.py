import cv2
import mediapipe as mp
import time
import csv
import os


# ==========================================
# CLASSE RILEVATORE EMOZIONI
# ==========================================

class RilevatoreEmozioni: # definisce una classe per creare oggetti

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, #modalità video
            max_num_faces=1, #massimo un volto per frame
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.file_csv = "dataset_emozioni.csv" #usando self. si creano deglia attributi 
        #dell'oggetto
        self.inizializza_csv() #chiama la funzione successiva, che crea il file se non esiste

    # --------------------------------------
    # CREA FILE CSV PER SALVARE I DATI
    # --------------------------------------

    def inizializza_csv(self): #creo una funzione della classe

        file_esiste = os.path.isfile(self.file_csv) #controlla se il file esiste e ritorna T/F

        with open(self.file_csv, mode='a', newline='') as file:
        #è una struttura built in di python per aprire un file in modo sicuro.usando solo open si dovrebbe 
        #chiudere il file alla fine, mentre with lo apre e chiude automatcamente. mode a vuol dire che si fa append, 
        #cioè si aggiunge senza cancellare
            writer = csv.writer(file) #crea un oggetto che scrive nel file csv

            if not file_esiste: #se il file non esiste
                writer.writerow([ #è una funzione di write, scrive una riga nel file CSV
                    "timestamp",
                    "punteggio_sorriso_0_100",
                    "apertura_bocca",
                    "occhio_sx",
                    "occhio_dx",
                    "emozione"
                ]) #tutte le colonne del dataset

    # --------------------------------------
    # SALVA RIGA NEL CSV
    # --------------------------------------

    def salva_dati(self, punteggio, apertura, occhio_sx, occhio_dx, emozione):

        timestamp = time.strftime("%H:%M:%S") 
        #salva l'ora corrente come stringa nel formato ore:minuti:secondi, è funzione della libreria time

        with open(self.file_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp,
                round(punteggio, 2),
                round(apertura, 4), #round(x, n) arrotonda il numero x a n cifre decimali
                round(occhio_sx, 4),
                round(occhio_dx, 4),
                emozione
            ])

    # --------------------------------------
    # CALCOLO SORRISO (NORMALIZZATO 0-100) USANDO SOGLIE EMPIRICHE
    # --------------------------------------

    def calcola_sorriso(self, landmarks):

        centro_y = landmarks[13].y #anzitutto si prendono i punti verticali della bocca
        angolo_sx_y = landmarks[61].y
        angolo_dx_y = landmarks[291].y

        diff_sx = centro_y - angolo_sx_y#si controlla se gli angoli della bocca 
        #sono più alti
        diff_dx = centro_y - angolo_dx_y

        punteggio_grezzo = 0 #punteggio di partenza nullo per l'espressione

        if diff_sx > 0 and diff_dx > 0:#se entrambi gli angoli sono sollevati
            #si aggiungono 20 punti
            punteggio_grezzo += 20

        punteggio_grezzo += (diff_sx + diff_dx) * 1000 #i valori di differenza sono
        #molto piccoli, quindi si moltiplica per 1000--> maggiori sono queste differenze
        #(angoli più sollevati), più intenso è il sorriso

        larghezza = abs(landmarks[291].x - landmarks[61].x) #calcola la larghezza della
        #bocca usando le coordinate x
        if larghezza > 0.15:
            punteggio_grezzo += 15 #se la bocca è molto larga si aggiungono altri punti

        apertura = abs(landmarks[14].y - landmarks[13].y) #calcola l'apertura 
        #della bocca (distanza verticale tra labbro superiore e inferiore)
        if apertura < 0.02:
            punteggio_grezzo += 10 #nei sorrisi naturali la bocca è poco aperta

        # 🔵 NORMALIZZAZIONE TRA 0 E 100
        punteggio_normalizzato = 50 + punteggio_grezzo 
        punteggio_normalizzato = max(0, min(100, punteggio_normalizzato))
        #così non possono esserci punteggi negativi, il centro viene spostato
        #su 50 (espressione neutra). si imposta così il limite tra o e 100

        return punteggio_normalizzato

    # --------------------------------------
    # ANALISI ESPRESSIONE
    # --------------------------------------

    def analizza(self, landmarks):

        punteggio = self.calcola_sorriso(landmarks)
        #ampiezza del sorriso trovata con un metodo separato 

        apertura = abs(landmarks[14].y - landmarks[13].y) #apertura delle labbra
        #se molto grande l'espressione viene classificata come sorpresa
        occhio_sx = abs(landmarks[145].y - landmarks[159].y) #trovo l'apertura dell'occhio 
        #considerando le palpebre a sx e a dx
        occhio_dx = abs(landmarks[374].y - landmarks[386].y)

        if apertura > 0.045:
            emozione = "SORPRESO" 

        elif punteggio > 75:
            emozione = "MOLTO FELICE" #sorriso molto ampio

        elif punteggio > 55:
            emozione = "FELICE"

        elif punteggio < 40:
            emozione = "ARRABBIATO" #sorriso assente o labbra serrate, il punteggio 
            #deve essere inferiore che rispetto al neutro (se arrabbiato gli angoli 
            #si spostano verso il basso)

        else:
            emozione = "NEUTRO" #nessuna delle condizioni precedenti si verifica

        return emozione, punteggio, apertura, occhio_sx, occhio_dx

    # --------------------------------------
    # AVVIO RILEVATORE
    # --------------------------------------

    def avvia(self):

        print("\n🎭 RILEVATORE EMOZIONI + DATASET")
        print("Premi 'q' per uscire\n") #stampa un messaggio di benvenuto sul terminale

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #apre la webcam del computer 
        #e usa directshow (solo su Windows, migliora la compatibilità)

        if not cap.isOpened():
            print("❌ Webcam non disponibile")
            return #se la webcam non viene aperta correttamente, stamoa un messaggio ed 
        #esce dal metodo

        print("✅ Webcam aperta")
        print("📁 Salvataggio dati in:", self.file_csv)

        try:
            while True:

                ret, frame = cap.read() #restituisce True se il frame è stato letto correttamente;
                #e restituisce anche l'immagine acquisita
                if not ret: #se la lettura fallisce, esce dal ciclo
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #si converte da BGR (OpenCV) a RGB (MediaPipe)
                results = self.face_mesh.process(rgb) #il modello FaceMesh rileva landmark sul volto

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks: #ciclo su ogni volto rilevato

                        emozione, punteggio, apertura, occhio_sx, occhio_dx = \
                            self.analizza(face_landmarks.landmark) #chiama il metodo analizza sui landmark del volto
 
                        self.salva_dati( #salva i dati trovati nel file CSV
                            punteggio,
                            apertura,
                            occhio_sx,
                            occhio_dx,
                            emozione
                        )

                        cv2.putText(frame, #scrive testo sul frame
                                    f"EMOZIONE: {emozione}", #testo
                                    (30, 50), #posizione (x, y)
                                    cv2.FONT_HERSHEY_SIMPLEX, #font
                                    1,#dimensione
                                    (0, 255, 0),#colore BGR
                                    3) #spessore

                        cv2.putText(frame,
                                    f"Sorriso (0-100): {punteggio:.1f}", #punteggio sorriso con una cifra decimale
                                    (30, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (255, 255, 255),
                                    2)

                cv2.imshow("Emotion Dataset Recorder", frame) #mostra la finestra della webcam 

                if cv2.waitKey(1) & 0xFF == ord('q'): #codice numerico del tasto q
                    break

        finally:
            cap.release() #libera la webcam
            cv2.destroyAllWindows() #chiude tutte le finestre aperte da OpenCV
            print("✅ Programma terminato")
#quelle con cv2 sono funzioni della libreria OpenCV, che serve a usare webcam, 
#elaborare immagini e riconoscere volti
#cap è invece una variabile scelta nel codice = cv2. VideoCapture (0) che contiene
#l'oggetto che controlla la webcam

# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 60) #ripete 60 volte =, riga decorativa
    print("🤖 SISTEMA RILEVAMENTO EMOZIONI + DATASET")
    print("=" * 60)

    rilevatore = RilevatoreEmozioni() #crea un'istanza del rilevatore
    rilevatore.avvia() #avvia il programma


if __name__ == "__main__": #esegue il codice solo se questo file è il principale
    main( )