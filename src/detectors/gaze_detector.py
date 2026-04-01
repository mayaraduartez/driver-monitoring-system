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