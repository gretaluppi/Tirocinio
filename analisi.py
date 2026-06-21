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

    emozione_stabile = applica_isteresi(emozione, stato, config)

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
    )
