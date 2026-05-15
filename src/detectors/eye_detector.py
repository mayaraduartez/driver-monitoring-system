from utils.math_utils import euclidean_distance


class EyeStateTracker:
    """
    Tarefa 1.1: Capturar estado do olho por frame
    
    - Usa EAR para determinar se olho está aberto ou fechado
    - Define threshold claro para classificação
    - Gera estado binário: EYE_OPEN ou EYE_CLOSED
    - Aplica suavização leve para reduzir jitter
    """
    
    def __init__(self, ear_closed_threshold=0.05, smoothing_window=3):
        """
        Args:
            ear_closed_threshold: Valor de EAR abaixo do qual olho é considerado fechado
            smoothing_window: Número de frames para suavização (janela móvel)
        """
        self.EAR_CLOSED_THRESHOLD = ear_closed_threshold
        self.smoothing_window = smoothing_window
        self.ear_history = []  # Histórico de EAR para suavização
        self.state_history = []  # Histórico de estados (0=fechado, 1=aberto)
        self.smoothed_ear = None
        self.current_state = None
    
    def update(self, ear_average):
        """
        Atualiza o estado do olho baseado na métrica EAR.
        
        Args:
            ear_average: Valor de EAR calculado do frame atual
            
        Returns:
            dict: {
                'ear_raw': float - EAR bruto do frame
                'ear_smoothed': float - EAR suavizado
                'state': str - 'EYE_OPEN' ou 'EYE_CLOSED'
                'state_numeric': int - 1 para aberto, 0 para fechado
            }
        """
        # Armazena EAR bruto no histórico
        self.ear_history.append(ear_average)
        
        # Mantém janela de histórico
        if len(self.ear_history) > self.smoothing_window:
            self.ear_history.pop(0)
        
        # Calcula média móvel para suavização
        self.smoothed_ear = sum(self.ear_history) / len(self.ear_history)
        
        # Determina estado: 1 = aberto, 0 = fechado
        state_numeric = 1 if self.smoothed_ear >= self.EAR_CLOSED_THRESHOLD else 0
        self.state_history.append(state_numeric)
        
        # Mantém histórico de estados
        if len(self.state_history) > self.smoothing_window:
            self.state_history.pop(0)
        
        # Define string de estado
        state_str = "EYE_OPEN" if state_numeric == 1 else "EYE_CLOSED"
        self.current_state = state_str
        
        return {
            'ear_raw': ear_average,
            'ear_smoothed': self.smoothed_ear,
            'state': state_str,
            'state_numeric': state_numeric
        }
    
    def get_continuous_closure_frames(self):
        """
        Retorna quantos frames consecutivos o olho está fechado.
        Útil para diferenciar piscada de fechamento prolongado.
        """
        if not self.state_history:
            return 0
        
        count = 0
        for state in reversed(self.state_history):
            if state == 0:  # fechado
                count += 1
            else:
                break
        return count
    
    def reset(self):
        """Reseta o histórico (usar quando face é perdida)."""
        self.ear_history = []
        self.state_history = []
        self.smoothed_ear = None
        self.current_state = None


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


def get_eye_aspect_ratio(face_landmarks):
    """
    Calcula EAR (Eye Aspect Ratio) para ambos os olhos.
    
    EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
    
    Onde P1-P6 são as landmarks ao redor do olho.
    Landmarks do olho direito: 33-46
    Landmarks do olho esquerdo: 263-276
    
    Returns:
        tuple: (ear_right, ear_left, ear_average)
    """
    
    # Landmarks do olho direito
    right_p1 = face_landmarks[33]  # canto esquerdo
    right_p2 = face_landmarks[37]  # pálpebra superior esquerda
    right_p3 = face_landmarks[38]  # pálpebra superior centro
    right_p4 = face_landmarks[40]  # canto direito
    right_p5 = face_landmarks[41]  # pálpebra inferior centro
    right_p6 = face_landmarks[42]  # pálpebra inferior esquerda
    
    # Landmarks do olho esquerdo
    left_p1 = face_landmarks[263]  # canto esquerdo
    left_p2 = face_landmarks[267]  # pálpebra superior esquerda
    left_p3 = face_landmarks[268]  # pálpebra superior centro
    left_p4 = face_landmarks[270]  # canto direito
    left_p5 = face_landmarks[271]  # pálpebra inferior centro
    left_p6 = face_landmarks[272]  # pálpebra inferior esquerda
    
    def calculate_ear(p1, p2, p3, p4, p5, p6):
        vertical_left = euclidean_distance(p2, p6)
        vertical_center = euclidean_distance(p3, p5)
        horizontal = euclidean_distance(p1, p4)
        
        ear = (vertical_left + vertical_center) / (2 * horizontal)
        return ear
    
    ear_right = calculate_ear(right_p1, right_p2, right_p3, right_p4, right_p5, right_p6)
    ear_left = calculate_ear(left_p1, left_p2, left_p3, left_p4, left_p5, left_p6)
    ear_average = (ear_right + ear_left) / 2
    
    return ear_right, ear_left, ear_average
