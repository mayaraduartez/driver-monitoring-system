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

def is_hand_on_eyes(
        
    hand_landmarks_list,
    left_eye_x,
    left_eye_y,
    right_eye_x,
    right_eye_y,
    threshold=0.08,
    min_points_near=2
):
    if not hand_landmarks_list:
        return False

    for hand_landmarks in hand_landmarks_list:
        points_near = 0

        for index in FINGER_POINTS:
            point = hand_landmarks[index]

            dist_left = math.sqrt(
                (point.x - left_eye_x) ** 2 +
                (point.y - left_eye_y) ** 2
            )

            dist_right = math.sqrt(
                (point.x - right_eye_x) ** 2 +
                (point.y - right_eye_y) ** 2
            )

            if dist_left < threshold or dist_right < threshold:
                points_near += 1

        if points_near >= min_points_near:
            return True

    return False
    
class HandBehaviorTracker:
    def __init__(
        self,
        increase_rate=0.04,
        decrease_rate=0.005,
        missing_tolerance_frames=10
    ):
        self.increase_rate = increase_rate
        self.decrease_rate = decrease_rate
        self.missing_tolerance_frames = missing_tolerance_frames

        self.mouth_occlusion_confidence = 0.0
        self.eye_occlusion_confidence = 0.0

        self.mouth_missing_frames = 0
        self.eye_missing_frames = 0

    def _update_confidence(self, current_confidence, missing_frames, evidence_present):
        if evidence_present:
            missing_frames = 0
            current_confidence += self.increase_rate
        else:
            missing_frames += 1

            if missing_frames > self.missing_tolerance_frames:
                current_confidence -= self.decrease_rate

        current_confidence = max(0.0, min(current_confidence, 1.0))

        return current_confidence, missing_frames

    def update(self, hand_on_mouth, hand_on_eyes):
        self.mouth_occlusion_confidence, self.mouth_missing_frames = self._update_confidence(
            self.mouth_occlusion_confidence,
            self.mouth_missing_frames,
            hand_on_mouth
        )

        self.eye_occlusion_confidence, self.eye_missing_frames = self._update_confidence(
            self.eye_occlusion_confidence,
            self.eye_missing_frames,
            hand_on_eyes
        )

        return {
            "mouth_occlusion_confidence": self.mouth_occlusion_confidence,
            "eye_occlusion_confidence": self.eye_occlusion_confidence
        }

    def reset(self):
        self.mouth_occlusion_confidence = 0.0
        self.eye_occlusion_confidence = 0.0
        self.mouth_missing_frames = 0
        self.eye_missing_frames = 0