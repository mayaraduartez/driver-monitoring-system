import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
import math
from utils.drawing import draw_landmarks_on_image
from utils.math_utils import euclidean_distance
from utils.metric_state import MetricState
from detectors.eye_detector import get_gaze_direction, get_eye_aspect_ratio
from detectors.mouth_detector import get_mouth_data
from detectors.hand_detector import is_hand_on_mouth

# Inicializacao 
capture = cv2.VideoCapture(0) # 0 ou o caminho do video 

# carrega o modelo de detecção facial do arquivo face_landmarker.task q foi baixado 
base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')

# configura as opções do modelo de detecção facial, como o número máximo de faces a serem detectadas e os limiares de confiança para detecção e rastreamento
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=2,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
# carrega o modelo de detecção facial usando as opções definidas acima
detector = vision.FaceLandmarker.create_from_options(options)

# Hands
base_options_hands = python.BaseOptions(model_asset_path='models/hand_landmarker.task')

options_hands = vision.HandLandmarkerOptions(
    base_options=base_options_hands,
    num_hands=2
)
hand_detector = vision.HandLandmarker.create_from_options(options_hands)

# ===== INICIALIZAR METRIC STATES =====
# Estados temporais para cada métrica
eye_closure_state = MetricState(
    name='eye_closure',
    gain_up=0.15,
    gain_down=0.05,
    ema_alpha=0.3,
    cooldown_seconds=1.0
)

yawn_state = MetricState(
    name='yawn',
    gain_up=0.20,
    gain_down=0.08,
    ema_alpha=0.3,
    cooldown_seconds=2.0
)

gaze_state = MetricState(
    name='gaze_distraction',
    gain_up=0.10,
    gain_down=0.06,
    ema_alpha=0.3,
    cooldown_seconds=0.5
)

# ===== FUNÇÃO AUXILIAR =====
def get_gaze_ratio_numeric(face_landmarks):
    """Retorna gaze_ratio como número (0-1)"""
    right_inner = face_landmarks[33]
    right_outer = face_landmarks[133]
    left_inner = face_landmarks[362]
    left_outer = face_landmarks[263]
    right_iris = face_landmarks[468]
    left_iris = face_landmarks[473]
    
    def get_ratio(inner, iris, outer):
        return euclidean_distance(inner, iris) / euclidean_distance(inner, outer)
    
    right_ratio = get_ratio(right_inner, right_iris, right_outer)
    left_ratio = get_ratio(left_inner, left_iris, left_outer)
    gaze_ratio = (right_ratio + left_ratio) / 2
    
    return gaze_ratio

# loop principal: lê os frames da câmera, processa as detecções faciais e de mãos, e exibe os resultados na tela
while True:
    # Lê um frame da câmera
    success, frame = capture.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)  # Espelhar o feed da câmera

    # converte o frame de BGR (formato padrão do OpenCV) para RGB (formato esperado pelo MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # realiza a detecção facial e de mãos usando os modelos carregados, e armazena os resultados em detection_result e hand_result
    detection_result = detector.detect(mp_image)
    hand_result = hand_detector.detect(mp_image)

    # tamanho do frame para calcular as posições relativas das landmarks
    frame_h, frame_w, _ = frame.shape

    # desenha as landmarks faciais e de mãos na imagem usando a função draw_landmarks_on_image, e armazena a imagem anotada em annotated_image
    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)

    # maos
    if hand_result.hand_landmarks:
        for hand_landmarks in hand_result.hand_landmarks:
            drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks,
                vision.HandLandmarksConnections.HAND_CONNECTIONS,
                drawing_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                drawing_utils.DrawingSpec(color=(255,255,255), thickness=2)
            )

    # lógica 
    face_detected = detection_result.face_landmarks and len(detection_result.face_landmarks) > 0

    if face_detected:
        for face_landmarks in detection_result.face_landmarks:
            # ===== EXTRAIR MÉTRICAS =====
            # EAR (Eye Aspect Ratio)
            ear_right, ear_left, ear_average = get_eye_aspect_ratio(face_landmarks)
            
            # MAR (Mouth Aspect Ratio)
            mar, mouth_x, mouth_y = get_mouth_data(face_landmarks)
            
            # Gaze Direction
            gaze_direction = get_gaze_direction(face_landmarks)
            
            # Hand on mouth
            mao_na_boca = is_hand_on_mouth(
                hand_result.hand_landmarks,
                mouth_x,
                mouth_y
            )
            
            # ===== CONVERTER PARA EVIDÊNCIA (0-1) =====
            # Evidência de olho fechado: EAR baixo = alta evidência
            eye_closure_evidence = max(0.0, 1.0 - (ear_average / 0.2))  # Normalizar para 0-1
            eye_closure_evidence = min(1.0, eye_closure_evidence)
            
            # Evidência de bocejo: MAR alto = evidência de bocejo
            yawn_evidence = max(0.0, (mar - 0.4) / 0.3) if mar > 0.4 else 0.0
            yawn_evidence = min(1.0, yawn_evidence)
            
            # Evidência de desatenção: olhando para os lados
            gaze_ratio = get_gaze_ratio_numeric(face_landmarks)  # 0-1
            gaze_distraction_evidence = 1.0 if (gaze_ratio < 0.35 or gaze_ratio > 0.65) else 0.0
            
            # ===== ATUALIZAR METRIC STATES =====
            eye_closure_state.update(eye_closure_evidence, is_available=True)
            yawn_state.update(yawn_evidence, is_available=True)
            gaze_state.update(gaze_distraction_evidence, is_available=True)
            
            # ===== DESENHAR INFORMAÇÕES NO FRAME =====
            # Cores para diferentes estados
            color_normal = (0, 255, 0)      # Verde
            color_warning = (0, 165, 255)   # Laranja
            color_alert = (0, 0, 255)       # Vermelho
            
            # Linha 1: Métricas brutas
            cv2.putText(annotated_image, f"EAR: {ear_average:.3f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(annotated_image, f"MAR: {mar:.3f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Linha 2: Confiança de Olho Fechado
            conf_eye = eye_closure_state.confidence
            color_eye = color_alert if conf_eye > 0.7 else color_warning if conf_eye > 0.4 else color_normal
            cv2.putText(annotated_image, f"Olho: {conf_eye:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_eye, 2)
            
            # Linha 3: Confiança de Bocejo
            conf_yawn = yawn_state.confidence
            color_yawn = color_alert if conf_yawn > 0.7 else color_warning if conf_yawn > 0.4 else color_normal
            cv2.putText(annotated_image, f"Bocejo: {conf_yawn:.2f}", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_yawn, 2)
            
            # Linha 4: Direção do Olhar
            cv2.putText(annotated_image, f"Gaze: {gaze_direction}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 150, 255), 1)
            
            # Mão na boca
            if mao_na_boca:
                cv2.putText(annotated_image, "Mao na boca", (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # ===== ALERTAS =====
            alert_y = 250
            
            # Alerta 1: Olho Fechado
            if conf_eye > 0.7 and not eye_closure_state.is_on_cooldown():
                cv2.putText(annotated_image, "!!! OLHO FECHADO !!!", (20, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                eye_closure_state.trigger_cooldown()
                alert_y += 50
            
            # Alerta 2: Bocejo/Sono
            if conf_yawn > 0.7 and not yawn_state.is_on_cooldown():
                cv2.putText(annotated_image, "!!! BOCEJO DETECTADO !!!", (20, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                yawn_state.trigger_cooldown()
                alert_y += 50
            
            # Alerta 3: Desatenção (gaze)
            if conf_yawn > 0.3 and conf_eye > 0.3:  # Combinação: bocejo + fechamento
                cv2.putText(annotated_image, ">>> POSSÍVEL SONOLÊNCIA <<<", (20, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2)
    
    else:
        # Face não detectada
        eye_closure_state.update(0.0, is_available=False)
        yawn_state.update(0.0, is_available=False)
        gaze_state.update(0.0, is_available=False)
        
        cv2.putText(annotated_image, "Face nao detectada", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

   
    cv2.imshow('Annotated Image', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()