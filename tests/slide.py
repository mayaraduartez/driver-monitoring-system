import cv2 #OpenCv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np 
import fitz  # PyMuPDF manipula arquivos pdf
import time
import tkinter as tk
from tkinter import filedialog # cria selecao de arquivos e interacao com o usuario

# function para selecionar o pdf
def selecionar_pdf():
    caminho = filedialog.askopenfilename( # permite ao usuario selecionar um arquivo
        title="Selecione um PDF",
        filetypes=[("Arquivos PDF", "*.pdf")]
    )
    return caminho

caminho_pdf = selecionar_pdf()

if not caminho_pdf:
    print("Nenhum arquivo selecionado.")
    exit()


#abre o pdc e informa a pagina atual
doc = fitz.open(caminho_pdf)
pagina_atual = 0

# def que renderiza a pagina especifica e converte para formato OpenCV
def render_page(num):
    page = doc.load_page(num) # abre a pagina desejada
    pix = page.get_pixmap() # renderiza a pagina como imagem
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n) # transforma a imagem em um array numpy
    
    # Converter para BGR (OpenCV)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img

# detectar mãos
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# contar dedos
def contar_dedos(hand):
    dedos = 0

    tips = [8, 12, 16, 20]

    for tip in tips:
        if hand[tip].y < hand[tip - 2].y:
            dedos += 1

    return dedos

# camera
capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Erro ao acessar câmera")
    exit()

# Controle de tempo
last_action = 0
delay = 1  # segundos

# LOOP PRINCIPAL
while True:
    success, frame = capture.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)

    # =========================
    # ✋ GESTOS
    # =========================
    if detection_result.hand_landmarks:
        hand = detection_result.hand_landmarks[0]
        dedos = contar_dedos(hand)

        if time.time() - last_action > delay:
            # 👉 1 dedo = próxima página
            if dedos == 1:
                pagina_atual = min(pagina_atual + 1, len(doc) - 1)
                print("Próxima página")
                last_action = time.time()

            # ✌️ 2 dedos = página anterior
            elif dedos == 2:
                pagina_atual = max(pagina_atual - 1, 0)
                print("Página anterior")
                last_action = time.time()

    # mostra pdf
    pdf_img = render_page(pagina_atual)

    # Redimensionar PDF pra caber na tela
    pdf_img = cv2.resize(pdf_img, (800, 1000))

    cv2.imshow("PDF", pdf_img)
    cv2.imshow("Camera", frame)

    # ESC para sair
    if cv2.waitKey(1) & 0xFF == 27:
        break

# liberar recursos
capture.release()
cv2.destroyAllWindows()