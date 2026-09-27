"""Schema unico del dataset e delle feature per il modello.

Tenere qui questi nomi evita che raccolta dati, training e inferenza usino
colonne diverse per errore.
"""

BLENDSHAPE_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
    "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight", "tongueOut",
]

BLENDSHAPE_COLUMNS = [f"blendshape_{nome}" for nome in BLENDSHAPE_NAMES]

BASE_COLUMNS = [
    "timestamp", "session_id", "codice_persona",
    "punteggio_sorriso_0_100", "apertura_bocca", "occhio_sx", "occhio_dx",
    "apertura_spalle", "inclinazione_spalle", "inclinazione_busto",
    "stato_posturale", "valence", "arousal",
    "head_yaw", "head_pitch", "head_roll", "head_pose_sorgente", "attenzione_schermo",
    "calibrazione_postura", "etichetta_reale", "emozione",
    "sorgente_emozione", "confidenza_modello",
]

INTESTAZIONE_CSV = BASE_COLUMNS + BLENDSHAPE_COLUMNS
MODEL_FEATURES = BLENDSHAPE_COLUMNS


def valori_blendshape(bs):
    """Restituisce sempre le 52 feature, nell'ordine previsto dal modello."""
    return {f"blendshape_{nome}": round(float(bs.get(nome, 0.0)), 6) for nome in BLENDSHAPE_NAMES}
