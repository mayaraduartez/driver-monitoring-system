import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
import numpy as np
import math

# functions de apoio
def euclidean_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# funcao que desenha as landmarks na imagem (pontos e linhas)
def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)

    if not detection_result.face_landmarks:
        return annotated_image

    for face_landmarks in detection_result.face_landmarks:

        # CONTORNO DO ROSTO
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=drawing_utils.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
            connection_drawing_spec=drawing_utils.DrawingSpec(color=(255,255,255), thickness=1)
        )

        # ÍRIS
        drawing_utils.draw_landmarks(
            annotated_image,
            face_landmarks,
            vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            None,
            drawing_utils.DrawingSpec(color=(0,255,255), thickness=1)
        )

        drawing_utils.draw_landmarks(
            annotated_image,
            face_landmarks,
            vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            None,
            drawing_utils.DrawingSpec(color=(0,255,255), thickness=1)
        )

    return annotated_image

# Inicializacao 
capture = cv2.VideoCapture(0) # 0 ou o caminho do video 

# carrega o modelo de detecção facial do arquivo face_landmarker.task q foi baixado 
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')

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
base_options_hands = python.BaseOptions(model_asset_path='hand_landmarker.task')

options_hands = vision.HandLandmarkerOptions(
    base_options=base_options_hands,
    num_hands=2
)
hand_detector = vision.HandLandmarker.create_from_options(options_hands)

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

            # ===== DIREÇÃO DO OLHAR (DOIS OLHOS) =====

            # Pontos de referência para os olhos
            # olho direito
            right_inner = face_landmarks[33]
            right_outer = face_landmarks[133]

            # olho esquerdo
            left_inner  = face_landmarks[362]
            left_outer  = face_landmarks[263]

            # Centros das íris
            right_iris = face_landmarks[468]
            left_iris  = face_landmarks[473]

            # converter as coordenadas normalizadas das landmarks para coordenadas de pixel, multiplicando as coordenadas normalizadas pelo tamanho do frame
            def to_px(lm):
                return int(lm.x * frame_w), int(lm.y * frame_h)

            r_iris_px = to_px(right_iris)
            l_iris_px = to_px(left_iris)

            # desenha círculos nos centros das íris para visualização
            cv2.circle(frame, r_iris_px, 3, (255, 0, 0), -1)
            cv2.circle(frame, l_iris_px, 3, (255, 0, 0), -1)

            # Razões: mede a posição relativa da íris em relação aos cantos interno e externo do olho, usando a distância euclidiana entre os pontos de referência e o centro da íris. Uma razão menor indica que a pessoa está olhando para a esquerda, enquanto uma razão maior indica que está olhando para a direita. Razões próximas a 0.5 indicam que a pessoa está olhando para a frente.
            def get_ratio(inner, iris, outer):
                return euclidean_distance(inner, iris) / euclidean_distance(inner, outer)

            # calcula as razões para ambos os olhos e tira a média para obter uma estimativa mais robusta da direção do olhar, reduzindo o impacto de possíveis erros de detecção em um dos olhos
            right_ratio = get_ratio(right_inner, right_iris, right_outer)
            left_ratio  = get_ratio(left_inner,  left_iris,  left_outer)
            gaze_ratio = (right_ratio + left_ratio) / 2

            # classificação
            if gaze_ratio < 0.4:
                direcao = "Olhando ESQUERDA"
            elif gaze_ratio > 0.6:
                direcao = "Olhando DIREITA"
            else:
                direcao = "Olhando FRENTE"

            # Boca 
            top_lip = face_landmarks[13]
            bottom_lip = face_landmarks[14]
            
            # calcula a posição média da boca usando as coordenadas do lábio superior e inferior, para determinar se a mão está próxima da boca
            mouth_x = (top_lip.x + bottom_lip.x) / 2
            mouth_y = (top_lip.y + bottom_lip.y) / 2

            # pontos de referência para os olhos usados no cálculo da razão da boca, para normalizar a distância entre os lábios em relação ao tamanho do rosto, usando a distância entre os olhos como referência
            left_eye = face_landmarks[33]
            right_eye = face_landmarks[263]

            
            mouth_dist = euclidean_distance(top_lip, bottom_lip)
            eye_dist = euclidean_distance(left_eye, right_eye)

            mouth_ratio = mouth_dist / eye_dist

            # Mão na boca
            mao_na_boca = False

            if hand_result.hand_landmarks:
                for hand_landmarks in hand_result.hand_landmarks:
                    # loop em cada ponto da mão
                    for point in hand_landmarks:
                        dist = math.sqrt((point.x - mouth_x)**2 + (point.y - mouth_y)**2)
                        if dist < 0.05:
                            mao_na_boca = True
                            break
                    if mao_na_boca:
                        break

            # exibe a direção do olhar e se a mão está na boca na imagem anotada usando a função cv2.putText, que desenha texto na imagem. A direção do olhar é exibida em azul, enquanto a indicação de bocejo com mão é exibida em vermelho.
            cv2.putText(annotated_image, direcao,
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2)

            if mao_na_boca:
                cv2.putText(annotated_image, "Mao na boca",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2)

   
    cv2.imshow('Annotated Image', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()