import math


FINGER_POINTS = [4, 8, 12, 16, 20]


def is_hand_on_mouth(
    hand_landmarks_list,
    mouth_x,
    mouth_y,
    threshold=0.08,
    min_points_near=2
):
    if not hand_landmarks_list:
        return False

    for hand_landmarks in hand_landmarks_list:
        points_near = 0

        for index in FINGER_POINTS:
            point = hand_landmarks[index]

            dist = math.sqrt(
                (point.x - mouth_x) ** 2 +
                (point.y - mouth_y) ** 2
            )

            if dist < threshold:
                points_near += 1

        if points_near >= min_points_near:
            return True

    return False