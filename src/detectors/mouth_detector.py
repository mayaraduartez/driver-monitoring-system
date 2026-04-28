from utils.math_utils import euclidean_distance

def get_mouth_data(face_landmarks):
    # Landmarks da boca (MediaPipe Face Landmarks)
    # Cantos: 11 (esquerdo), 16 (direito)
    # Lábio superior: 12 (esquerdo), 13 (centro), 17 (direito)
    # Lábio inferior: 15 (esquerdo), 14 (centro), 18 (direito)
    
    mouth_left = face_landmarks[11]
    mouth_right = face_landmarks[16]
    
    top_left = face_landmarks[12]
    top_center = face_landmarks[13]
    top_right = face_landmarks[17]
    
    bottom_left = face_landmarks[15]
    bottom_center = face_landmarks[14]
    bottom_right = face_landmarks[18]
    
    # Calcular distâncias verticais
    vertical_left = euclidean_distance(top_left, bottom_left)
    vertical_center = euclidean_distance(top_center, bottom_center)
    vertical_right = euclidean_distance(top_right, bottom_right)
    
    # Calcular distância horizontal
    horizontal = euclidean_distance(mouth_left, mouth_right)
    
    # MAR (Mouth Aspect Ratio)
    mar = (vertical_left + vertical_center + vertical_right) / (2 * horizontal)
    
    # Coordenadas centrais da boca
    mouth_x = (top_center.x + bottom_center.x) / 2
    mouth_y = (top_center.y + bottom_center.y) / 2
    
    return mar, mouth_x, mouth_y