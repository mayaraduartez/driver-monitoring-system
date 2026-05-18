import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
import math
from utils.drawing import draw_landmarks_on_image
from utils.math_utils import euclidean_distance
from detectors.eye_detector import get_gaze_direction, get_eye_aspect_ratio, EyeStateTracker, PerclosTracker
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


#Inicializar rastreador de estado do olho
eye_state_tracker = EyeStateTracker(
    smoothing_window=5,
    calibration_frames=60,
    closed_ratio=0.72,
    open_margin=0.03
)
# inicializa a janela de tempo do perclos
perclos_tracker = PerclosTracker(window_seconds=20)




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
    limiar = 0.5

    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            direcao = get_gaze_direction(face_landmarks)

            mouth_ratio, mouth_x, mouth_y = get_mouth_data(face_landmarks)

            mao_na_boca = is_hand_on_mouth(
                hand_result.hand_landmarks,
                mouth_x,
                mouth_y
            )
            
            #Atualizar estado do olho com EAR
            ear_right, ear_left, ear_average = get_eye_aspect_ratio(face_landmarks)
            eye_state = eye_state_tracker.update(ear_average)
            perclos_data = perclos_tracker.update(eye_state["state_numeric"])

            # confianca perclos
            closed_confidence = eye_state["confidence"]

            # exibe a direção do olhar e se a mão está na boca na imagem anotada usando a função cv2.putText, que desenha texto na imagem. A direção do olhar é exibida em azul, enquanto a indicação de bocejo com mão é exibida em vermelho.
            cv2.putText(annotated_image, direcao,
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2)
            
            # Exibe estado do olho e EAR suavizado
            eye_color = (0, 255, 0) if eye_state['state'] == 'EYE_OPEN' else (0, 0, 255)
            cv2.putText(
                annotated_image,
                f"Olho: {eye_state['state']} | EAR: {eye_state['ear_smoothed']:.3f} | TH: {eye_state['threshold']:.3f}",
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                eye_color,
                2
            )

            cv2.putText(
                annotated_image,
                f"PERCLOS: {perclos_data['perclos']:.1f}% | Fechados: {perclos_data['closed_frames']}/{perclos_data['total_frames']}",
                (50, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                annotated_image,
                f"Conf olho fechado: {closed_confidence:.2f}",
                (50, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            if mao_na_boca:
                cv2.putText(annotated_image, "Mao na boca",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2)
    else:
        # Quando face não é detectada, resetar o rastreador de estado
        perclos_tracker.reset()

   
    cv2.imshow('Annotated Image', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()