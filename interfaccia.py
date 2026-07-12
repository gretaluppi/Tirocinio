import cv2


COLORI_EMOZIONI = {
    "MOLTO FELICE": (0, 215, 255),
    "FELICE": (80, 200, 120),
    "NEUTRO": (220, 220, 220),
    "ARRABBIATO": (70, 70, 255),
    "SORPRESO": (255, 170, 70),
    "TESO": (120, 180, 255),
    "SERENO": (120, 220, 180),
}


def colore_emozione(emozione):
    return COLORI_EMOZIONI.get(emozione, (255, 255, 255))


def acquisisci_consenso_privacy(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            return False

        frame = cv2.flip(frame, 1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (35, 35), (690, 255), (15, 18, 30), -1)
        cv2.rectangle(overlay, (35, 35), (690, 255), (0, 215, 255), 2)
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

        righe = [
            ("CONSENSO ALLA RACCOLTA DATI", 0.85, (240, 240, 240), 2),
            ("Il sistema non salva immagini o video della webcam.", 0.58, (220, 220, 220), 1),
            ("Vengono salvate solo metriche numeriche derivate.", 0.58, (220, 220, 220), 1),
            ("Il codice persona deve essere anonimo.", 0.58, (220, 220, 220), 1),
            ("Premi C per continuare  |  ESC per annullare", 0.62, (0, 215, 255), 1),
        ]

        y = 78
        for testo, scala, colore, spessore in righe:
            cv2.putText(frame, testo, (60, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scala, colore, spessore, cv2.LINE_AA)
            y += 38

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
        cv2.rectangle(overlay, (35, 35), (605, 215), (15, 18, 30), -1)
        cv2.rectangle(overlay, (35, 35), (605, 215), (0, 215, 255), 2)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, "INSERISCI CODICE PERSONA", (60, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(frame, "Digita il codice e premi INVIO per confermare", (60, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, "BACKSPACE cancella  |  ESC esce", (60, 146),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        box_color = (0, 215, 255) if codice else (120, 120, 120)
        cv2.rectangle(frame, (60, 165), (340, 200), box_color, 2)
        cv2.putText(frame, codice if codice else "_", (72, 190),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

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
                     calibrazione_stato="DISATTIVATA", calibrazione_progresso=1.0):
    overlay = frame.copy()
    colore = colore_emozione(emozione)
    h, w, _ = frame.shape

    cv2.rectangle(overlay, (18, 18), (390, 226), (15, 18, 30), -1)
    cv2.rectangle(overlay, (18, 18), (390, 226), colore, 2)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, "EMOTIONAL MIRRORING", (30, 42),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(frame, emozione, (30, 73),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, colore, 2, cv2.LINE_AA)

    barra_x, barra_y = 30, 86
    barra_w, barra_h = 180, 12
    riempimento = int((max(0, min(100, punteggio)) / 100) * barra_w)
    cv2.rectangle(frame, (barra_x, barra_y),
                  (barra_x + barra_w, barra_y + barra_h), (90, 90, 90), 1)
    cv2.rectangle(frame, (barra_x, barra_y),
                  (barra_x + riempimento, barra_y + barra_h), colore, -1)

    cv2.putText(frame, f"Sorriso {punteggio:04.1f}", (30, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Bocca {apertura:.3f}", (145, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (205, 205, 205), 1, cv2.LINE_AA)
    cv2.putText(frame, stato_posturale, (30, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (215, 215, 215), 1, cv2.LINE_AA)
    cv2.putText(frame, f"V {valence:+.2f}  A {arousal:.2f}", (30, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (215, 215, 215), 1, cv2.LINE_AA)
    attenzione_testo = "ATTENTO" if attenzione else "SGUARDO NON CENTRATO"
    cv2.putText(frame, attenzione_testo, (145, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (205, 205, 205), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Calibrazione {calibrazione_stato} {calibrazione_progresso * 100:03.0f}%",
                (30, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (205, 205, 205), 1, cv2.LINE_AA)
    etichetta = stato.get("etichetta_reale", "") or "NON IMPOSTATA"
    cv2.putText(frame, f"Training: {etichetta}", (30, 208),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (205, 205, 205), 1, cv2.LINE_AA)
    cv2.putText(frame, f"ID {stato['codice_persona']}", (215, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q/ESC esci | D debug blendshapes", (w - 370, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)


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
                  (x0 + panel_w, y0 + panel_h), (15, 18, 30), -1)
    cv2.rectangle(overlay, (x0, y0),
                  (x0 + panel_w, y0 + panel_h), (100, 100, 100), 1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "BLENDSHAPES", (x0 + 10, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 220, 220), 1, cv2.LINE_AA)
        barra_len = int(max(0, min(1, valore)) * 82)
        barra_x = col_x + 150
        cv2.rectangle(frame, (barra_x, y - 8),
                      (barra_x + 82, y - 1), (85, 85, 85), 1)
        cv2.rectangle(frame, (barra_x, y - 8),
                      (barra_x + barra_len, y - 1), (0, 215, 255), -1)
