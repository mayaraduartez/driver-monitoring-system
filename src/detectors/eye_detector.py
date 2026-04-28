from utils.math_utils import euclidean_distance

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
