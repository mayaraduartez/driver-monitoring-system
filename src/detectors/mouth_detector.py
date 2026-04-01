from utils.math_utils import euclidean_distance

def get_mouth_data(face_landmarks):
    top_lip = face_landmarks[13]
    bottom_lip = face_landmarks[14]

    mouth_x = (top_lip.x + bottom_lip.x) / 2
    mouth_y = (top_lip.y + bottom_lip.y) / 2

    left_eye = face_landmarks[33]
    right_eye = face_landmarks[263]

    mouth_dist = euclidean_distance(top_lip, bottom_lip)
    eye_dist = euclidean_distance(left_eye, right_eye)

    mouth_ratio = mouth_dist / eye_dist

    return mouth_ratio, mouth_x, mouth_y