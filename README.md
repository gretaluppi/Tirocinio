# Emotional Mirroring - versione finale

Applicazione locale che legge una webcam e registra **solo metriche derivate** del volto e della postura. Non salva immagini o video.

## Avvio rapido

1. Installare Python 3.10+ e, nella cartella del progetto, eseguire `pip install -r requirements.txt`.
2. Verificare che `face_landmarker.task` e `pose_landmarker_full.task` siano nella cartella principale.
3. Avviare `python finalcode.py`.
4. Confermare il consenso, inserire un codice anonimo e usare i tasti `1`-`7` per indicare l'etichetta reale durante la raccolta dati.

I dati finiscono nella cartella `dati/`. Il CSV e' sempre prodotto; il Parquet viene creato alla chiusura se `raccolta_dati.scrivi_parquet` e' attivo e sono installati pandas e pyarrow.

## Addestramento del modello

Dopo aver raccolto almeno 100 righe etichettate da almeno due persone, eseguire:

```bash
python train_model.py
```

Il comando genera `modello_emozioni.pkl` e `report_training.json`. Al successivo avvio `finalcode.py` carica automaticamente il modello. Una predizione ML viene usata solo se la sua confidenza supera `ml.confidenza_minima`; altrimenti resta attiva la classificazione euristica.

## Struttura

- `finalcode.py`: avvio, webcam e ciclo realtime.
- `analisi.py`: filtri, postura, valence/arousal e classificazione euristica.
- `schema_dati.py`: le 52 feature blendshape e intestazione del dataset.
- `registratore_dati.py`: CSV e Parquet opzionale.
- `train_model.py` e `modello_ml.py`: training e inferenza ML.
- `interfaccia.py`, `realtime_server.py`, `dashboard/`: pannello OpenCV e dashboard WebSocket locale.
- `config.json`: soglie e opzioni modificabili senza cambiare codice.

La relazione tecnica completa e' in `Relazione_finale_Emotional_Mirroring.docx`.
