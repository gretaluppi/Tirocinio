import re

import cv2


# Formato ammesso per il codice persona: lettera P + 3 cifre (es. P001).
# Cambia questa espressione regolare se vuoi un altro formato pseudonimo.
CODICE_PERSONA_PATTERN = re.compile(r"^P\d{3}$")


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


def scala_interfaccia(frame, larghezza_progetto, altezza_progetto, margine=18):
    """Riduce i pannelli su webcam piccole, senza farli uscire dal frame."""
    altezza, larghezza = frame.shape[:2]
    return max(0.45, min(1.0, (larghezza - 2 * margine) / larghezza_progetto,
                         (altezza - 2 * margine) / altezza_progetto))


def acquisisci_consenso_privacy(cap, conservazione_giorni=None):
    durata_testo = f"massimo {conservazione_giorni} giorni" if conservazione_giorni else "vedi PRIVACY.md"

    righe = [
        ("CONSENSO ALLA RACCOLTA DATI", 0.85, (240, 240, 240), 2),
        ("Cosa: metriche numeriche di espressioni e postura.", 0.55, (220, 220, 220), 1),
        ("Il sistema NON salva immagini o video della webcam.", 0.55, (220, 220, 220), 1),
        ("Perche': analisi per un progetto di tirocinio.", 0.55, (220, 220, 220), 1),
        ("Dove: solo su questo computer, cartella locale 'dati'.", 0.55, (220, 220, 220), 1),
        (f"Conservazione: {durata_testo}.", 0.55, (220, 220, 220), 1),
        ("Cancellazione: puoi chiederla in qualsiasi momento.", 0.55, (220, 220, 220), 1),
        ("Il codice persona deve essere anonimo (es. P001).", 0.55, (220, 220, 220), 1),
        ("Premi C per continuare  |  ESC per annullare", 0.62, (0, 215, 255), 1),
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            return False

        frame = cv2.flip(frame, 1)
        overlay = frame.copy()
        panel_w_progetto, panel_h_progetto = 700, 60 + len(righe) * 34
        s = scala_interfaccia(frame, panel_w_progetto, panel_h_progetto, 35)
        h, w = frame.shape[:2]
        panel_w, panel_h = int(panel_w_progetto * s), int(panel_h_progetto * s)
        x0, y0 = (w - panel_w) // 2, (h - panel_h) // 2
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (15, 18, 30), -1)
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 215, 255), max(1, int(2 * s)))
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

        y = y0 + int(40 * s)
        for testo, scala, colore, spessore in righe:
            cv2.putText(frame, testo, (x0 + int(25 * s), y), cv2.FONT_HERSHEY_SIMPLEX,
                        scala * s, colore, max(1, int(spessore * s)), cv2.LINE_AA)
            y += int(34 * s)

        cv2.imshow("Emotion Dataset Recorder", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return False
        if key in (ord("c"), ord("C")):
            return True


def acquisisci_codice_persona_da_camera(cap):
    codice = ""
    errore_formato = False

    while True:
        ret, frame = cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        overlay = frame.copy()
        s = scala_interfaccia(frame, 570, 205, 35)
        h, w = frame.shape[:2]
        panel_w, panel_h = int(570 * s), int(205 * s)
        x0, y0 = (w - panel_w) // 2, (h - panel_h) // 2
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (15, 18, 30), -1)
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 215, 255), max(1, int(2 * s)))
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, "INSERISCI CODICE PERSONA", (x0 + int(25 * s), y0 + int(45 * s)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9 * s, (240, 240, 240), max(1, int(2 * s)), cv2.LINE_AA)
        cv2.putText(frame, "Formato richiesto: P001 (lettera P + 3 cifre)", (x0 + int(25 * s), y0 + int(83 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (220, 220, 220), max(1, int(s)), cv2.LINE_AA)
        cv2.putText(frame, "BACKSPACE cancella  |  ESC esce", (x0 + int(25 * s), y0 + int(108 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (200, 200, 200), max(1, int(s)), cv2.LINE_AA)

        box_color = (0, 215, 255) if codice else (120, 120, 120)
        cv2.rectangle(frame, (x0 + int(25 * s), y0 + int(122 * s)),
                      (x0 + int(305 * s), y0 + int(157 * s)), box_color, max(1, int(2 * s)))
        cv2.putText(frame, codice if codice else "_", (x0 + int(37 * s), y0 + int(147 * s)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8 * s, (255, 255, 255), max(1, int(s)), cv2.LINE_AA)

        if errore_formato:
            cv2.putText(frame, "Formato non valido, riprova (es. P001)", (x0 + int(25 * s), y0 + int(182 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (70, 70, 255), max(1, int(s)), cv2.LINE_AA)

        cv2.imshow("Emotion Dataset Recorder", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return None
        if key in (13, 10):
            codice_pulito = codice.strip().upper()
            if CODICE_PERSONA_PATTERN.match(codice_pulito):
                return codice_pulito
            errore_formato = True
        elif key == 8:
            codice = codice[:-1]
            errore_formato = False
        elif 32 <= key <= 126 and len(codice) < 20:
            codice += chr(key)
            errore_formato = False


def disegna_pannello(frame, emozione, punteggio, apertura, stato_posturale, stato,
                     valence=0.0, arousal=0.0, attenzione=False,
                     calibrazione_stato="DISATTIVATA", calibrazione_progresso=1.0,
                     sorgente_emozione="EURISTICA", confidenza_modello=0.0):
    overlay = frame.copy()
    colore = colore_emozione(emozione)
    h, w, _ = frame.shape
    s = scala_interfaccia(frame, 392, 232)
    x0, y0 = 18, 18

    cv2.rectangle(overlay, (x0, y0), (x0 + int(392 * s), y0 + int(232 * s)), (15, 18, 30), -1)
    cv2.rectangle(overlay, (x0, y0), (x0 + int(392 * s), y0 + int(232 * s)), colore, max(1, int(2 * s)))
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, "EMOTIONAL MIRRORING", (x0 + int(12 * s), y0 + int(24 * s)),
                cv2.FONT_HERSHEY_DUPLEX, 0.52 * s, (240, 240, 240), max(1, int(s)), cv2.LINE_AA)
    cv2.putText(frame, emozione, (x0 + int(12 * s), y0 + int(55 * s)),
                cv2.FONT_HERSHEY_DUPLEX, 0.8 * s, colore, max(1, int(2 * s)), cv2.LINE_AA)

    barra_x, barra_y = x0 + int(12 * s), y0 + int(68 * s)
    barra_w, barra_h = int(180 * s), max(2, int(12 * s))
    riempimento = int((max(0, min(100, punteggio)) / 100) * barra_w)
    cv2.rectangle(frame, (barra_x, barra_y),
                  (barra_x + barra_w, barra_y + barra_h), (90, 90, 90), 1)
    cv2.rectangle(frame, (barra_x, barra_y),
                  (barra_x + riempimento, barra_y + barra_h), colore, -1)

    cv2.putText(frame, f"Sorriso {punteggio:04.1f}", (x0 + int(12 * s), y0 + int(94 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48 * s, (245, 245, 245), max(1, int(s)), cv2.LINE_AA)
    cv2.putText(frame, f"Bocca {apertura:.3f}", (x0 + int(127 * s), y0 + int(94 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, (205, 205, 205), max(1, int(s)), cv2.LINE_AA)
    cv2.putText(frame, stato_posturale, (x0 + int(12 * s), y0 + int(117 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46 * s, (215, 215, 215), max(1, int(s)), cv2.LINE_AA)
    cv2.putText(frame, f"V {valence:+.2f}  A {arousal:.2f}", (x0 + int(12 * s), y0 + int(142 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, (215, 215, 215), max(1, int(s)), cv2.LINE_AA)
    attenzione_testo = "ATTENTO" if attenzione else "SGUARDO NON CENTRATO"
    cv2.putText(frame, attenzione_testo, (x0 + int(127 * s), y0 + int(142 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42 * s, (205, 205, 205), max(1, int(s)), cv2.LINE_AA)
    cv2.putText(frame, f"Calibrazione {calibrazione_stato} {calibrazione_progresso * 100:03.0f}%",
                (x0 + int(12 * s), y0 + int(166 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * s, (205, 205, 205), max(1, int(s)), cv2.LINE_AA)
    fonte = sorgente_emozione if sorgente_emozione != "ML" else f"ML {confidenza_modello:.0%}"
    cv2.putText(frame, f"Classificazione: {fonte}", (x0 + int(12 * s), y0 + int(187 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42 * s, (205, 205, 205), max(1, int(s)), cv2.LINE_AA)
    etichetta = stato.get("etichetta_reale", "") or "NON IMPOSTATA"
    cv2.putText(frame, f"Training: {etichetta}", (x0 + int(12 * s), y0 + int(209 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42 * s, (205, 205, 205), max(1, int(s)), cv2.LINE_AA)
    cv2.putText(frame, f"ID {stato['codice_persona']}", (x0 + int(197 * s), y0 + int(24 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42 * s, (210, 210, 210), max(1, int(s)), cv2.LINE_AA)
    cv2.putText(frame, "Q/ESC esci | D debug blendshapes", (max(18, w - int(370 * s)), h - max(12, int(20 * s))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (230, 230, 230), max(1, int(s)), cv2.LINE_AA)


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
