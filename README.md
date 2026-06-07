# Driver Monitoring System

Sistema de monitoramento de condutores baseado em Visão Computacional para detecção de sonolência e distração em tempo real.

O sistema utiliza OpenCV e MediaPipe para análise facial e comportamental, combinando múltiplas evidências para estimar o risco de fadiga ou desatenção durante a condução.

---

## Funcionalidades

### Sonolência

- Detecção de fechamento ocular (EAR)
- Cálculo de PERCLOS
- Detecção de bocejo (MAR)
- Detecção de bocejo ocultado por mão na boca
- Detecção de mão nos olhos
- Detecção de apoio da cabeça (mão no rosto)
- Acumuladores temporais de confiança

### Distração

- Rastreamento da direção do olhar
- Detecção de olhar fora da via
- Detecção de padrão compatível com uso de celular
- Fusão de evidências comportamentais

### Processamento de imagem

- Correção adaptativa de gama
- Equalização de histograma
- Redução de ruído
- Realce de nitidez

### Alertas

- Alerta sonoro de sonolência
- Alerta sonoro de distração
- Alerta crítico

---

## Arquitetura

```text
Câmera
   │
   ▼
Pré-processamento
   │
   ▼
MediaPipe
(Face + Hands)
   │
   ▼
Detectores
 ├── Olhos
 ├── Boca
 └── Mãos
   │
   ▼
Acumuladores Temporais
   │
   ▼
Alert Manager
   │
   ▼
Alerta Sonoro
```

---

## Estrutura do Projeto

```text
landmark-ai/
│
├── assets/
├── models/
├── sounds/
│
├── src/
│   ├── detectors/
│   │   ├── alert_manager.py
│   │   ├── eye_detector.py
│   │   ├── hand_detector.py
│   │   └── mouth_detector.py
│   │
│   ├── utils/
│   │   ├── drawing.py
│   │   ├── math_utils.py
│   │   └── preprocessing.py
│   │
│   └── main.py
│
├── experiments/
├── README.md
└── requirements.txt
```

---

## Tecnologias Utilizadas

- Python 3.11+
- OpenCV
- MediaPipe Tasks
- NumPy

---

## Instalação

Clone o repositório:

```bash
git clone https://github.com/mayaraduartez/driver-monitoring-system.git
cd driver-monitoring-system
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

## Execução

```bash
python src/main.py
```

---

## Modelos Utilizados

O sistema utiliza:

- Face Landmarker (MediaPipe)
- Hand Landmarker (MediaPipe)

Arquivos:

```text
models/
├── face_landmarker.task
└── hand_landmarker.task
```

---

## Autor

Mayara Duarte

