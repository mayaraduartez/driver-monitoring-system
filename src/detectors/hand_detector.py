import math

def is_hand_on_mouth(hand_landmarks_list, mouth_x, mouth_y, threshold=0.08):
    if not hand_landmarks_list:
        return False

    for hand_landmarks in hand_landmarks_list:
        for point in hand_landmarks:
            dist = math.sqrt((point.x - mouth_x)**2 + (point.y - mouth_y)**2)

            if dist < threshold:
                return True

    return False