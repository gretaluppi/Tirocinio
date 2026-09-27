import cv2


COLORI_EMOZIONI = {
    "MOLTO FELICE": (255, 200, 50),
    "FELICE": (100, 220, 140),
    "NEUTRO": (200, 200, 210),
    "ARRABBIATO": (235, 80, 80),
    "SORPRESO": (255, 160, 80),
    "TESO": (140, 160, 230),
    "SERENO": (100, 200, 180),
}

# Palette colori UI moderna
COLORI_UI = {
    "sfondo": (18, 22, 28),
    "sfondo_light": (28, 34, 42),
    "testo_primario": (245, 247, 250),
    "testo_secondario": (180, 188, 200),
    "testo_muted": (140, 150, 165),
    "accento": (80, 200, 240),
    "successo": (100, 220, 140),
    "attenzione": (255, 180, 80),
    "errore": (235, 80, 80),
}


def colore_emozione(emozione):
    return COLORI_EMOZIONI.get(emozione, (255, 255, 255))


def scala_interfaccia(frame, larghezza_progetto, altezza_progetto, margine=18):
    """Riduce i pannelli su webcam piccole, senza farli uscire dal frame."""
    altezza, larghezza = frame.shape[:2]
    return max(0.45, min(1.0, (larghezza - 2 * margine) / larghezza_progetto,
                         (altezza - 2 * margine) / altezza_progetto))


def acquisisci_consenso_privacy(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            return False

        frame = cv2.flip(frame, 1)
        overlay = frame.copy()
        s = scala_interfaccia(frame, 655, 220, 35)
        h, w = frame.shape[:2]
        panel_w, panel_h = int(655 * s), int(220 * s)
        x0, y0 = (w - panel_w) // 2, (h - panel_h) // 2
        
        # Design migliorato con gradiente
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), COLORI_UI["sfondo"], -1)
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + int(50 * s)), COLORI_UI["sfondo_light"], -1)
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), COLORI_UI["accento"], max(1, int(2 * s)))
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        righe = [
            ("CONSENSO ALLA RACCOLTA DATI", 0.85, COLORI_UI["testo_primario"], 2),
            ("Il sistema non salva immagini o video della webcam.", 0.58, COLORI_UI["testo_secondario"], 1),
            ("Vengono salvate solo metriche numeriche derivate.", 0.58, COLORI_UI["testo_secondario"], 1),
            ("Il codice persona deve essere anonimo.", 0.58, COLORI_UI["testo_secondario"], 1),
            ("Premi C per continuare  |  ESC per annullare", 0.62, COLORI_UI["accento"], 1),
        ]

        y = y0 + int(43 * s)
        for testo, scala, colore, spessore in righe:
            cv2.putText(frame, testo, (x0 + int(25 * s), y), cv2.FONT_HERSHEY_SIMPLEX,
                        scala * s, colore, max(1, int(spessore * s)), cv2.LINE_AA)
            y += int(38 * s)

        cv2.imshow("Emotion Dataset Recorder", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return False
        if key in (ord("c"), ord("C")):
            return True


def acquisisci_codice_persona_da_camera(cap):
    codice = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        overlay = frame.copy()
        s = scala_interfaccia(frame, 570, 180, 35)
        h, w = frame.shape[:2]
        panel_w, panel_h = int(570 * s), int(180 * s)
        x0, y0 = (w - panel_w) // 2, (h - panel_h) // 2
        
        # Design migliorato
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), COLORI_UI["sfondo"], -1)
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), COLORI_UI["accento"], max(1, int(2 * s)))
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        cv2.putText(frame, "INSERISCI CODICE PERSONA", (x0 + int(25 * s), y0 + int(45 * s)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9 * s, COLORI_UI["testo_primario"], max(1, int(2 * s)), cv2.LINE_AA)
        cv2.putText(frame, "Digita il codice e premi INVIO per confermare", (x0 + int(25 * s), y0 + int(83 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65 * s, COLORI_UI["testo_secondario"], max(1, int(s)), cv2.LINE_AA)
        cv2.putText(frame, "BACKSPACE cancella  |  ESC esce", (x0 + int(25 * s), y0 + int(111 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, COLORI_UI["testo_muted"], max(1, int(s)), cv2.LINE_AA)

        box_color = COLORI_UI["accento"] if codice else COLORI_UI["testo_muted"]
        cv2.rectangle(frame, (x0 + int(25 * s), y0 + int(130 * s)),
                      (x0 + int(305 * s), y0 + int(165 * s)), box_color, max(1, int(2 * s)))
        cv2.putText(frame, codice if codice else "_", (x0 + int(37 * s), y0 + int(155 * s)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8 * s, COLORI_UI["testo_primario"], max(1, int(s)), cv2.LINE_AA)

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


def disegna_pannello(frame, emozione, punteggio, apertura, stato_posturale, stato,
                     valence=0.0, arousal=0.0, attenzione=False,
                     calibrazione_stato="DISATTIVATA", calibrazione_progresso=1.0,
                     sorgente_emozione="EURISTICA", confidenza_modello=0.0):
    overlay = frame.copy()
    colore = colore_emozione(emozione)
    h, w, _ = frame.shape
    s = scala_interfaccia(frame, 420, 260)
    x0, y0 = 18, 18
    panel_w, panel_h = int(420 * s), int(260 * s)

    # Sfondo con gradiente simulato
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), COLORI_UI["sfondo"], -1)
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + int(40 * s)), COLORI_UI["sfondo_light"], -1)
    
    # Bordo colorato in alto
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + int(3 * s)), colore, -1)
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), colore, max(1, int(1.5 * s)))
    
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Header
    cv2.putText(frame, "EMOTIONAL MIRRORING", (x0 + int(16 * s), y0 + int(26 * s)),
                cv2.FONT_HERSHEY_DUPLEX, 0.55 * s, COLORI_UI["testo_primario"], max(1, int(1.2 * s)), cv2.LINE_AA)
    
    # Emozione principale
    cv2.putText(frame, emozione, (x0 + int(16 * s), y0 + int(58 * s)),
                cv2.FONT_HERSHEY_DUPLEX, 0.95 * s, colore, max(1, int(2.5 * s)), cv2.LINE_AA)

    # Barra sorriso migliorata
    barra_x, barra_y = x0 + int(16 * s), y0 + int(72 * s)
    barra_w, barra_h = int(200 * s), max(3, int(14 * s))
    riempimento = int((max(0, min(100, punteggio)) / 100) * barra_w)
    cv2.rectangle(frame, (barra_x, barra_y),
                  (barra_x + barra_w, barra_y + barra_h), (60, 65, 75), -1)
    cv2.rectangle(frame, (barra_x, barra_y),
                  (barra_x + riempimento, barra_y + barra_h), colore, -1)

    # Metriche principali
    cv2.putText(frame, f"Sorriso {punteggio:05.1f}", (x0 + int(16 * s), y0 + int(102 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52 * s, COLORI_UI["testo_primario"], max(1, int(1.2 * s)), cv2.LINE_AA)
    cv2.putText(frame, f"Bocca {apertura:.3f}", (x0 + int(140 * s), y0 + int(102 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48 * s, COLORI_UI["testo_secondario"], max(1, int(s)), cv2.LINE_AA)
    
    # Postura con indicatore visivo
    colore_postura = COLORI_UI["successo"] if "APERTA" in stato_posturale else (COLORI_UI["errore"] if "CHIUSA" in stato_posturale else COLORI_UI["testo_muted"])
    cv2.putText(frame, stato_posturale, (x0 + int(16 * s), y0 + int(128 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, colore_postura, max(1, int(1.2 * s)), cv2.LINE_AA)
    
    # Valence/Arousal con colori
    colore_valence = COLORI_UI["successo"] if valence > 0.1 else (COLORI_UI["errore"] if valence < -0.1 else COLORI_UI["testo_secondario"])
    cv2.putText(frame, f"V {valence:+.2f}", (x0 + int(16 * s), y0 + int(155 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48 * s, colore_valence, max(1, int(s)), cv2.LINE_AA)
    cv2.putText(frame, f"A {arousal:.2f}", (x0 + int(80 * s), y0 + int(155 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48 * s, COLORI_UI["attenzione"], max(1, int(s)), cv2.LINE_AA)
    
    # Attenzione
    colore_attenzione = COLORI_UI["successo"] if attenzione else COLORI_UI["errore"]
    attenzione_testo = "ATTENTO" if attenzione else "NON CENTRATO"
    cv2.putText(frame, attenzione_testo, (x0 + int(140 * s), y0 + int(155 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46 * s, colore_attenzione, max(1, int(s)), cv2.LINE_AA)
    
    # Calibrazione con barra
    cv2.putText(frame, f"Calibrazione {calibrazione_stato}", (x0 + int(16 * s), y0 + int(182 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44 * s, COLORI_UI["testo_secondario"], max(1, int(s)), cv2.LINE_AA)
    calib_x, calib_y = x0 + int(16 * s), y0 + int(192 * s)
    calib_w, calib_h = int(120 * s), max(2, int(6 * s))
    cv2.rectangle(frame, (calib_x, calib_y), (calib_x + calib_w, calib_y + calib_h), (60, 65, 75), -1)
    cv2.rectangle(frame, (calib_x, calib_y), (calib_x + int(calib_w * calibrazione_progresso), calib_y + calib_h), COLORI_UI["accento"], -1)
    
    # Info classificazione
    fonte = sorgente_emozione if sorgente_emozione != "ML" else f"ML {confidenza_modello:.0%}"
    cv2.putText(frame, f"{fonte}", (x0 + int(16 * s), y0 + int(215 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42 * s, COLORI_UI["testo_muted"], max(1, int(s)), cv2.LINE_AA)
    
    # Training
    etichetta = stato.get("etichetta_reale", "") or "-"
    cv2.putText(frame, f"T: {etichetta}", (x0 + int(140 * s), y0 + int(215 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42 * s, COLORI_UI["testo_muted"], max(1, int(s)), cv2.LINE_AA)
    
    # ID sessione in alto a destra
    cv2.putText(frame, f"ID {stato['codice_persona']}", (x0 + int(220 * s), y0 + int(26 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42 * s, COLORI_UI["testo_muted"], max(1, int(s)), cv2.LINE_AA)
    
    # Shortcut in basso
    cv2.putText(frame, "Q/ESC esci | D debug", (max(18, w - int(160 * s)), h - max(14, int(24 * s))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44 * s, COLORI_UI["testo_muted"], max(1, int(s)), cv2.LINE_AA)


def disegna_debug_blendshapes(frame, bs):
    chiavi = [
        "mouthSmileLeft", "mouthSmileRight", "jawOpen",
        "browInnerUp", "browDownLeft", "browDownRight",
        "eyeBlinkLeft", "eyeBlinkRight",
        "mouthFrownLeft", "mouthFrownRight",
        "eyeSquintLeft", "eyeSquintRight",
        "mouthPressLeft", "mouthPressRight",
    ]
    h, w, _ = frame.shape
    righe = (len(chiavi) + 1) // 2
    panel_w = min(w - 36, 604)
    panel_h = 34 + righe * 20
    x0 = max(18, w - panel_w - 18)
    y0 = 220 if w < 760 else 25
    if y0 + panel_h > h - 42:
        y0 = max(18, h - panel_h - 42)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0),
                  (x0 + panel_w, y0 + panel_h), COLORI_UI["sfondo"], -1)
    cv2.rectangle(overlay, (x0, y0),
                  (x0 + panel_w, y0 + panel_h), COLORI_UI["accento"], 1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.putText(frame, "BLENDSHAPES", (x0 + 10, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORI_UI["testo_secondario"], 1, cv2.LINE_AA)

    for i, chiave in enumerate(chiavi):
        valore = bs.get(chiave, 0)
        colonna = i // righe
        riga = i % righe
        col_w = panel_w // 2
        col_x = x0 + 10 + colonna * col_w
        y = y0 + 40 + riga * 20
        nome_corto = (
            chiave.replace("mouth", "m")
            .replace("brow", "b")
            .replace("eye", "e")
        )
        cv2.putText(frame, f"{nome_corto}: {valore:.3f}", (col_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLORI_UI["testo_primario"], 1, cv2.LINE_AA)
        barra_len = int(max(0, min(1, valore)) * 82)
        barra_x = col_x + 150
        cv2.rectangle(frame, (barra_x, y - 8),
                      (barra_x + 82, y - 1), (60, 65, 75), 1)
        cv2.rectangle(frame, (barra_x, y - 8),
                      (barra_x + barra_len, y - 1), COLORI_UI["accento"], -1)
