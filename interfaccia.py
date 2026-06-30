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
                     valence=0.0, arousal=0.0, attenzione=False):
    overlay = frame.copy()
    colore = colore_emozione(emozione)
    h, w, _ = frame.shape

    cv2.rectangle(overlay, (18, 18), (360, 178), (15, 18, 30), -1)
    cv2.rectangle(overlay, (18, 18), (360, 178), colore, 2)
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
    ]
    h, w, _ = frame.shape
    x0 = w - 310
    y0 = 25

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 10, y0 - 18),
                  (w - 10, y0 + len(chiavi) * 22 + 5), (15, 18, 30), -1)
    cv2.rectangle(overlay, (x0 - 10, y0 - 18),
                  (w - 10, y0 + len(chiavi) * 22 + 5), (100, 100, 100), 1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "BLENDSHAPES", (x0, y0 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    for i, chiave in enumerate(chiavi):
        valore = bs.get(chiave, 0)
        y = y0 + 18 + i * 22
        nome_corto = chiave.replace("mouth", "m").replace("brow", "b").replace("eye", "e")
        cv2.putText(frame, f"{nome_corto}: {valore:.3f}", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        barra_len = int(valore * 120)
        cv2.rectangle(frame, (x0 + 170, y - 8),
                      (x0 + 170 + barra_len, y - 1), (0, 215, 255), -1)
