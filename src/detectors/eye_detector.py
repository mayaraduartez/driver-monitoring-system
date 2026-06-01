from utils.math_utils import euclidean_distance
from collections import deque
import time

# Calcula se o olho está aberto ou fechado usando EAR.
# Calcula para onde a pessoa está olhando: esquerda, direita ou frente.
# Calcula o valor eAR dos dois olhos

class EyeStateTracker:
    def __init__(
        self,
        smoothing_window=5, # quantos frames usados pra suavizar o EAR
        calibration_frames=60,# quantos frames usados pra calibrar o olho aberto
        closed_ratio=0.72, #percentual do EAR aberto que será considerado limite para olho fechado
        open_margin=0.03 #margem para evitar oscilação entre aberto e fechado (histerese)
    ):
        self.smoothing_window = smoothing_window
        self.calibration_frames = calibration_frames
        self.closed_ratio = closed_ratio
        self.open_margin = open_margin

        self.ear_history = [] # lista com ultimos valores de EAR
        self.open_ear_samples = [] #lista usada somente durante a calibraçao inicial

        self.open_ear = None # ear medio do olho aberto
        self.close_threshold = None
        self.open_threshold = None

        self.current_state = "CALIBRATING" #estado inicial

        self.closed_confidence = TemporalConfidence(
            increase_rate=0.08,
            decrease_rate=0.04
        )

    def update(self, ear_average):
        self.ear_history.append(ear_average)

        if len(self.ear_history) > self.smoothing_window:
            self.ear_history.pop(0)

        ear_smoothed = sum(self.ear_history) / len(self.ear_history)

        # calibração inicial assumindo que o usuário começa com olho aberto
        if len(self.open_ear_samples) < self.calibration_frames:
            self.open_ear_samples.append(ear_smoothed)

            self.open_ear = sum(self.open_ear_samples) / len(self.open_ear_samples) # calcula a media do olho aberto
            self.close_threshold = self.open_ear * self.closed_ratio # calcula limite do olho fechado
            self.open_threshold = self.close_threshold + self.open_margin # calcula o limite para voltar a considerar o olho aberto

            self.current_state = "CALIBRATING"

            return {
                "ear_raw": ear_average,
                "ear_smoothed": ear_smoothed,
                "state": self.current_state,
                "state_numeric": 1,
                "threshold": self.close_threshold,
                "open_ear": self.open_ear,
                "confidence": 0.0

            }

        # atualiza lentamente o EAR aberto quando o olho está claramente aberto
        if ear_smoothed > self.open_threshold:
            self.open_ear = (self.open_ear * 0.98) + (ear_smoothed * 0.02) # atualiza lentamente o valor de EAR aberto, usa 98% do antigo ee 4% do nov, para não mudar bruscamente
            #recalcula os thresholds
            self.close_threshold = self.open_ear * self.closed_ratio
            self.open_threshold = self.close_threshold + self.open_margin

        # histerese: evita ficar alternando aberto/fechado
        if self.current_state != "EYE_CLOSED":
            if ear_smoothed < self.close_threshold:
                self.current_state = "EYE_CLOSED"
            else:
                self.current_state = "EYE_OPEN"
        else:
            if ear_smoothed > self.open_threshold:
                self.current_state = "EYE_OPEN"

        confidence = self.closed_confidence.update(
            self.current_state == "EYE_CLOSED"
        )

        return {
            "ear_raw": ear_average, # ear original
            "ear_smoothed": ear_smoothed, # ear suavizado
            "state": self.current_state, # estado
            "state_numeric": 1 if self.current_state == "EYE_OPEN" else 0, # estado em 0 ou 1
            "threshold": self.close_threshold, # threshold de fechamento
            "open_ear": self.open_ear, # ear medio estimado para olho aberto 
            "confidence": confidence
        }

    # usado quando perde o rosto
    def reset(self):
        self.ear_history = []
        self.open_ear_samples = []
        self.open_ear = None
        self.close_threshold = None
        self.open_threshold = None
        self.current_state = "CALIBRATING"
        self.closed_confidence.reset()

class PerclosTracker:
    def __init__(self, window_seconds=20):
        self.window_seconds = window_seconds
        self.frames = deque()

    def update(self, eye_state_numeric):
        now = time.time()

        # Guarda o frame atual
        # 1 = olho aberto
        # 0 = olho fechado
        self.frames.append({
            "time": now,
            "closed": 1 if eye_state_numeric == 0 else 0
        })

        # Remove frames fora da janela de tempo
        while self.frames and now - self.frames[0]["time"] > self.window_seconds:
            self.frames.popleft()

        total_frames = len(self.frames)

        if total_frames == 0:
            return {
                "perclos": 0,
                "closed_frames": 0,
                "total_frames": 0
            }

        closed_frames = sum(frame["closed"] for frame in self.frames)

        perclos = (closed_frames / total_frames) * 100

        return {
            "perclos": perclos,
            "closed_frames": closed_frames,
            "total_frames": total_frames
        }

    def reset(self):
        self.frames.clear()

class TemporalConfidence:
    def __init__(
        self,
        increase_rate=0.08,
        decrease_rate=0.04,
        min_confidence=0.0,
        max_confidence=1.0
    ):
        self.increase_rate = increase_rate
        self.decrease_rate = decrease_rate
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.confidence = 0.0

    def update(self, evidence_present):
        """
        evidence_present:
            True  = evidência presente
            False = evidência ausente
        """

        if evidence_present:
            self.confidence += self.increase_rate
        else:
            self.confidence -= self.decrease_rate

        self.confidence = max(
            self.min_confidence,
            min(self.confidence, self.max_confidence)
        )

        return self.confidence

    def reset(self):
        self.confidence = 0.0

def get_gaze_direction(face_landmarks):
    right_inner = face_landmarks[33]
    right_outer = face_landmarks[133]

    left_inner  = face_landmarks[362]
    left_outer  = face_landmarks[263]

    right_iris = face_landmarks[468]
    left_iris  = face_landmarks[473]

    def get_ratio(inner, iris, outer):
        return euclidean_distance(inner, iris) / euclidean_distance(inner, outer)

    right_ratio = get_ratio(right_inner, right_iris, right_outer)
    left_ratio  = get_ratio(left_inner, left_iris, left_outer)

    gaze_ratio = (right_ratio + left_ratio) / 2

    if gaze_ratio < 0.4:
        return "Olhando ESQUERDA"
    elif gaze_ratio > 0.6:
        return "Olhando DIREITA"
    else:
        return "Olhando FRENTE"

# calcula o EAR dos dois olhos 
def get_eye_aspect_ratio(face_landmarks):
    right = [33, 160, 158, 133, 153, 144]
    left  = [362, 385, 387, 263, 373, 380]

    def calc(points):
        p1, p2, p3, p4, p5, p6 = [face_landmarks[i] for i in points]

        vertical_1 = euclidean_distance(p2, p6)
        vertical_2 = euclidean_distance(p3, p5)
        horizontal = euclidean_distance(p1, p4)

        if horizontal == 0:
            return 0

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    ear_right = calc(right)
    ear_left = calc(left)

    return ear_right, ear_left, (ear_right + ear_left) / 2

def get_safe_gaze_direction(face_landmarks, eye_state):
    if eye_state["state"] == "EYE_CLOSED":
        return "Olho fechado"

    if eye_state["state"] == "CALIBRATING":
        return "Calibrando olho"

    return get_gaze_direction(face_landmarks)
