import math
import time


# =============================================================================
# ONE EURO FILTER
# Filtro adattivo: stabile quando il segnale e' fermo, reattivo quando si muove.
# Riferimento: Casiez et al., "1euro filter: A Simple Speed-based Low-pass
# Filter for Noisy Input in Interactive Systems" (2012).
#
# Funzionamento: il filtro calcola la velocita' del segnale (derivata).
# Quando la velocita' e' bassa (segnale fermo), usa un cutoff basso che
# smussa molto. Quando la velocita' e' alta (movimento reale), alza il
# cutoff e lascia passare il cambiamento senza ritardo.
#
# Parametri:
#   min_cutoff: cutoff minimo (Hz) — piu' basso = piu' stabile a riposo
#   beta: sensibilita' alla velocita' — piu' alto = piu' reattivo ai movimenti
#   d_cutoff: cutoff per il filtro della derivata
# =============================================================================

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filtra(self, x, t=None):
        if t is None:
            t = time.time()

        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev

        a_d = self._alpha(self.d_cutoff, dt)
        dx = (x - self.x_prev) / dt
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


def crea_filtri(config):
    oef = config["one_euro_filter"]
    nomi = [
        "punteggio", "apertura", "occhio_sx", "occhio_dx",
        "apertura_spalle", "inclinazione_spalle", "inclinazione_busto",
        "apertura_spalle_3d", "inclinazione_spalle_3d", "inclinazione_busto_3d",
        "head_yaw", "head_pitch", "valence", "arousal",
    ]
    return {
        nome: OneEuroFilter(oef["min_cutoff"], oef["beta"], oef["d_cutoff"])
        for nome in nomi
    }


# =============================================================================
# BLENDSHAPES
# =============================================================================

def estrai_blendshapes(face_blendshapes):
    return {bs.category_name: bs.score for bs in face_blendshapes}


def limita(valore, minimo, massimo):
    return max(minimo, min(massimo, valore))


def inizializza_calibrazione_postura(config):
    return {
        "abilitata": config.get("calibrazione_postura", {}).get("abilitata", True),
        "durata": config.get("calibrazione_postura", {}).get("durata_secondi", 4.0),
        "inizio": None,
        "campioni": [],
        "baseline": None,
        "completata": False,
    }


def aggiorna_calibrazione_postura(calibrazione, apertura_spalle, inclinazione_spalle,
                                  inclinazione_busto, t):
    if not calibrazione or not calibrazione["abilitata"]:
        return

    if calibrazione["inizio"] is None:
        calibrazione["inizio"] = t

    if calibrazione["completata"]:
        return

    calibrazione["campioni"].append((apertura_spalle, inclinazione_spalle, inclinazione_busto))
    if t - calibrazione["inizio"] < calibrazione["durata"]:
        return

    n = len(calibrazione["campioni"])
    if n == 0:
        return

    calibrazione["baseline"] = {
        "apertura_spalle": sum(c[0] for c in calibrazione["campioni"]) / n,
        "inclinazione_spalle": sum(c[1] for c in calibrazione["campioni"]) / n,
        "inclinazione_busto": sum(c[2] for c in calibrazione["campioni"]) / n,
    }
    calibrazione["completata"] = True


def stato_calibrazione_postura(calibrazione, t):
    if not calibrazione or not calibrazione["abilitata"]:
        return "DISATTIVATA", 1.0
    if calibrazione["completata"]:
        return "COMPLETATA", 1.0
    if calibrazione["inizio"] is None:
        return "IN ATTESA", 0.0
    progresso = limita((t - calibrazione["inizio"]) / calibrazione["durata"], 0.0, 1.0)
    return "IN CORSO", progresso


# =============================================================================
# METRICHE FACCIALI
# Legge i blendshapes e li filtra con One Euro Filter.
# =============================================================================

def stabilizza_metriche(bs, filtri, t):
    smile = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
    punteggio = smile * 100

    apertura = bs.get("jawOpen", 0)
    occhio_sx = 1.0 - bs.get("eyeBlinkLeft", 0)
    occhio_dx = 1.0 - bs.get("eyeBlinkRight", 0)

    return (
        filtri["punteggio"].filtra(punteggio, t),
        filtri["apertura"].filtra(apertura, t),
        filtri["occhio_sx"].filtra(occhio_sx, t),
        filtri["occhio_dx"].filtra(occhio_dx, t),
    )


# =============================================================================
# POSTURA
# =============================================================================

def analizza_postura(pose_landmarks, filtri, config, t):
    spalla_sx = pose_landmarks[11]
    spalla_dx = pose_landmarks[12]
    anca_sx = pose_landmarks[23]
    anca_dx = pose_landmarks[24]
    naso = pose_landmarks[0]

    centro_spalle_x = (spalla_sx.x + spalla_dx.x) / 2
    centro_spalle_y = (spalla_sx.y + spalla_dx.y) / 2
    centro_anche_x = (anca_sx.x + anca_dx.x) / 2

    apertura_spalle = abs(spalla_dx.x - spalla_sx.x)
    inclinazione_spalle = abs(spalla_dx.y - spalla_sx.y)
    inclinazione_busto = abs(centro_spalle_x - centro_anche_x)
    testa_avanti = max(0.0, naso.y - centro_spalle_y)

    ap_s = filtri["apertura_spalle"].filtra(apertura_spalle, t)
    inc_s = filtri["inclinazione_spalle"].filtra(inclinazione_spalle, t)
    inc_b = filtri["inclinazione_busto"].filtra(inclinazione_busto, t)

    sp = config["soglie_postura"]
    stato_posturale = "POSTURA NEUTRA"
    if (
        ap_s < sp["apertura_spalle_chiusa"]
        or inc_b > sp["inclinazione_busto_chiusa"]
        or testa_avanti > sp["testa_avanti_chiusa"]
    ):
        stato_posturale = "POSTURA CHIUSA"
    elif (
        ap_s > sp["apertura_spalle_aperta"]
        and inc_s < sp["inclinazione_spalle_aperta"]
        and inc_b < sp["inclinazione_busto_aperta"]
    ):
        stato_posturale = "POSTURA APERTA"

    return ap_s, inc_s, inc_b, stato_posturale


def analizza_postura_3d(pose_world_landmarks, filtri, config, t, calibrazione=None):
    spalla_sx = pose_world_landmarks[11]
    spalla_dx = pose_world_landmarks[12]
    anca_sx = pose_world_landmarks[23]
    anca_dx = pose_world_landmarks[24]
    naso = pose_world_landmarks[0]

    def distanza(a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

    centro_spalle_x = (spalla_sx.x + spalla_dx.x) / 2
    centro_spalle_y = (spalla_sx.y + spalla_dx.y) / 2
    centro_spalle_z = (spalla_sx.z + spalla_dx.z) / 2
    centro_anche_x = (anca_sx.x + anca_dx.x) / 2
    centro_anche_y = (anca_sx.y + anca_dx.y) / 2
    centro_anche_z = (anca_sx.z + anca_dx.z) / 2

    apertura_spalle = distanza(spalla_sx, spalla_dx)
    inclinazione_spalle = math.degrees(math.atan2(spalla_dx.y - spalla_sx.y, spalla_dx.x - spalla_sx.x))
    dx = centro_spalle_x - centro_anche_x
    dy = centro_spalle_y - centro_anche_y
    dz = centro_spalle_z - centro_anche_z
    inclinazione_busto = math.degrees(math.atan2(math.sqrt(dx * dx + dz * dz), abs(dy) + 1e-6))
    testa_avanti = max(0.0, naso.z - centro_spalle_z)

    ap_s = filtri["apertura_spalle_3d"].filtra(apertura_spalle, t)
    inc_s = filtri["inclinazione_spalle_3d"].filtra(abs(inclinazione_spalle), t)
    inc_b = filtri["inclinazione_busto_3d"].filtra(inclinazione_busto, t)

    aggiorna_calibrazione_postura(calibrazione, ap_s, inc_s, inc_b, t)

    sp = config["soglie_postura_3d"]
    baseline = calibrazione.get("baseline") if calibrazione else None
    if baseline:
        cfg_cal = config["calibrazione_postura"]
        chiusa = (
            ap_s < baseline["apertura_spalle"] * cfg_cal["fattore_spalle_chiuse"]
            or inc_b > baseline["inclinazione_busto"] + cfg_cal["delta_busto_chiuso_gradi"]
            or testa_avanti > sp["testa_avanti_chiusa"]
        )
        aperta = (
            ap_s > baseline["apertura_spalle"] * cfg_cal["fattore_spalle_aperte"]
            and inc_s < baseline["inclinazione_spalle"] + cfg_cal["delta_spalle_aperte_gradi"]
            and inc_b < baseline["inclinazione_busto"] + cfg_cal["delta_busto_aperto_gradi"]
        )
    else:
        chiusa = (
            ap_s < sp["apertura_spalle_chiusa"]
            or inc_b > sp["inclinazione_busto_chiusa_gradi"]
            or testa_avanti > sp["testa_avanti_chiusa"]
        )
        aperta = (
            ap_s > sp["apertura_spalle_aperta"]
            and inc_s < sp["inclinazione_spalle_aperta_gradi"]
            and inc_b < sp["inclinazione_busto_aperta_gradi"]
        )

    stato_posturale = "POSTURA NEUTRA"
    if chiusa:
        stato_posturale = "POSTURA CHIUSA"
    elif aperta:
        stato_posturale = "POSTURA APERTA"

    return ap_s, inc_s, inc_b, stato_posturale


def stima_head_pose(bs, filtri, config, t):
    yaw = bs.get("eyeLookOutLeft", 0) - bs.get("eyeLookOutRight", 0)
    yaw += bs.get("eyeLookInRight", 0) - bs.get("eyeLookInLeft", 0)
    pitch = bs.get("eyeLookDownLeft", 0) + bs.get("eyeLookDownRight", 0)
    pitch -= bs.get("eyeLookUpLeft", 0) + bs.get("eyeLookUpRight", 0)

    yaw = filtri["head_yaw"].filtra(yaw, t)
    pitch = filtri["head_pitch"].filtra(pitch, t)
    soglie = config["attenzione"]
    attenzione = (
        abs(yaw) <= soglie["yaw_massimo"]
        and abs(pitch) <= soglie["pitch_massimo"]
    )

    return yaw, pitch, attenzione


def calcola_valence_arousal(bs, punteggio, apertura, brow_up, brow_down,
                            stato_posturale, filtri, config, t):
    frown = (bs.get("mouthFrownLeft", 0) + bs.get("mouthFrownRight", 0)) / 2
    smile = punteggio / 100
    postura = config["circumplex"]["postura"]

    valence = (smile * 2) - 1
    valence -= brow_down * 0.55
    valence -= frown * 0.75

    arousal = apertura * 1.6
    arousal += brow_up * 0.75
    arousal += brow_down * 0.45

    if stato_posturale == "POSTURA CHIUSA":
        valence += postura["chiusa_valence"]
        arousal += postura["chiusa_arousal"]
    elif stato_posturale == "POSTURA APERTA":
        valence += postura["aperta_valence"]
        arousal += postura["aperta_arousal"]

    valence = filtri["valence"].filtra(limita(valence, -1.0, 1.0), t)
    arousal = filtri["arousal"].filtra(limita(arousal, 0.0, 1.0), t)
    return valence, arousal


def etichetta_da_circumplex(valence, arousal, emozione_rule_based, config):
    soglie = config["circumplex"]["soglie"]

    if valence >= soglie["valence_positiva"] and arousal <= soglie["arousal_basso"]:
        return "SERENO"
    if valence <= soglie["valence_negativa"] and arousal >= soglie["arousal_alto"]:
        return "TESO"
    if valence >= soglie["valence_positiva"] and arousal >= soglie["arousal_alto"]:
        return "MOLTO FELICE" if emozione_rule_based == "MOLTO FELICE" else "FELICE"
    if emozione_rule_based == "SORPRESO" and arousal >= soglie["arousal_alto"]:
        return "SORPRESO"
    if emozione_rule_based == "ARRABBIATO" and valence <= soglie["valence_neutra_bassa"]:
        return "ARRABBIATO"

    return emozione_rule_based


# =============================================================================
# ISTERESI
# L'emozione cambia solo se la nuova candidata persiste per un tempo minimo.
# Evita oscillazioni rapide tra etichette vicine.
# =============================================================================

def applica_isteresi(emozione_nuova, stato, config):
    durata = config["isteresi"]["durata_minima"]
    adesso = time.time()

    if emozione_nuova != stato["emozione_candidata"]:
        stato["emozione_candidata"] = emozione_nuova
        stato["tempo_candidata"] = adesso

    if adesso - stato["tempo_candidata"] >= durata:
        stato["emozione_confermata"] = stato["emozione_candidata"]

    return stato["emozione_confermata"]


# =============================================================================
# CLASSIFICAZIONE EMOZIONE
# =============================================================================

def classifica_emozione(bs, filtri, stato, config, t, pose_info=None):
    punteggio, apertura, occhio_sx, occhio_dx = stabilizza_metriche(
        bs, filtri, t
    )

    brow_up = bs.get("browInnerUp", 0)
    brow_down = (bs.get("browDownLeft", 0) + bs.get("browDownRight", 0)) / 2

    soglie = config["soglie_emozioni"]

    if apertura > soglie["sorpreso_apertura"] and brow_up > soglie["sorpreso_sopracciglia"]:
        emozione = "SORPRESO"
    elif punteggio > soglie["molto_felice"]:
        emozione = "MOLTO FELICE"
    elif punteggio > soglie["felice"]:
        emozione = "FELICE"
    elif brow_down > soglie["arrabbiato_sopracciglia"] and punteggio < soglie["arrabbiato_sorriso"]:
        emozione = "ARRABBIATO"
    else:
        emozione = "NEUTRO"

    apertura_spalle = 0.0
    inclinazione_spalle = 0.0
    inclinazione_busto = 0.0
    stato_posturale = stato["ultimo_stato_posturale"]

    if pose_info is not None:
        apertura_spalle, inclinazione_spalle, inclinazione_busto, stato_posturale = pose_info
        stato["ultimo_stato_posturale"] = stato_posturale

        if stato_posturale == "POSTURA CHIUSA" and emozione in ("NEUTRO", "FELICE"):
            emozione = "TESO"
        elif stato_posturale == "POSTURA APERTA" and emozione == "NEUTRO":
            emozione = "SERENO"

    valence, arousal = calcola_valence_arousal(
        bs, punteggio, apertura, brow_up, brow_down, stato_posturale, filtri, config, t
    )
    emozione = etichetta_da_circumplex(valence, arousal, emozione, config)
    emozione_stabile = applica_isteresi(emozione, stato, config)
    head_yaw, head_pitch, attenzione = stima_head_pose(bs, filtri, config, t)

    return (
        emozione_stabile,
        punteggio,
        apertura,
        occhio_sx,
        occhio_dx,
        apertura_spalle,
        inclinazione_spalle,
        inclinazione_busto,
        stato_posturale,
        valence,
        arousal,
        head_yaw,
        head_pitch,
        attenzione,
    )
