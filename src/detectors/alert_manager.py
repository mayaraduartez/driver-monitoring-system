class DriverAlertManager:
    def __init__(
        self,
        sleepiness_alert_threshold=0.75,
        distraction_alert_threshold=0.75,
        critical_sleepiness_threshold=0.70,
        critical_distraction_threshold=0.60,
        attention_threshold=0.50
    ):
        self.sleepiness_alert_threshold = sleepiness_alert_threshold
        self.distraction_alert_threshold = distraction_alert_threshold
        self.critical_sleepiness_threshold = critical_sleepiness_threshold
        self.critical_distraction_threshold = critical_distraction_threshold
        self.attention_threshold = attention_threshold

    def update(
        self,
        eye_closed_confidence,
        perclos,
        mouth_yawn_confidence,
        gaze_confidence,
        is_phone_like_gaze,
        hand_on_mouth
    ):
        # Normaliza PERCLOS:
        # 0%  -> 0.0
        # 40% -> 1.0
        perclos_confidence = min(perclos / 40.0, 1.0)

        phone_like_confidence = 1.0 if is_phone_like_gaze else 0.0
        hand_confidence = 1.0 if hand_on_mouth else 0.0

        sleepiness_score = (
            eye_closed_confidence * 0.40 +
            perclos_confidence * 0.35 +
            mouth_yawn_confidence * 0.25
        )

        distraction_score = (
            gaze_confidence * 0.65 +
            phone_like_confidence * 0.25 +
            hand_confidence * 0.10
        )

        if (
            sleepiness_score >= self.critical_sleepiness_threshold
            and distraction_score >= self.critical_distraction_threshold
        ):
            level = "CRITICO"
            message = "Sonolencia e distracao detectadas"

        elif sleepiness_score >= self.sleepiness_alert_threshold:
            level = "ALERTA_SONOLENCIA"
            message = "Sinais de sonolencia detectados"

        elif distraction_score >= self.distraction_alert_threshold:
            level = "ALERTA_DISTRACAO"
            message = "Sinais de distracao detectados"

        elif (
            sleepiness_score >= self.attention_threshold
            or distraction_score >= self.attention_threshold
        ):
            level = "ATENCAO"
            message = "Sinais leves detectados"

        else:
            level = "NORMAL"
            message = "Conducao normal"

        return {
            "level": level,
            "message": message,
            "sleepiness_score": sleepiness_score,
            "distraction_score": distraction_score,
            "perclos_confidence": perclos_confidence,
            "phone_like_confidence": phone_like_confidence,
            "hand_confidence": hand_confidence
        }