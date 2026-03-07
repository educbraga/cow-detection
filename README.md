# WIP 👷‍♂️🏗️

# Cows Challenge 🐄 — Keypoints + Identificação (YOLO Pose)

Projeto de visão computacional para **detecção de keypoints em vacas (vista superior)** com **YOLO Pose (Ultralytics)**, extração de **features geométricas** e **classificação/identificação** de cada vaca.

<p align="center">
  <img src="./docs/logo.png" alt="Logo do projeto" width="700" />
</p>

## Objetivos

1. **Anotar** keypoints das vacas em imagens (top-view).
2. **Treinar e avaliar** um modelo **YOLO Pose**.
3. **Gerar features** a partir dos keypoints (ângulos, distâncias, proporções) que possam identificar cada animal.
4. **Analisar** descritivamente as features e sua utilidade.
5. **Treinar um classificador** usando as features (identificação da vaca).
6. **Avaliar** o modelo final.

---

## Dataset (visão geral)

- **30 vacas** na área de ordenha (_parlor milking station area_).
- **50 imagens por vaca**.
- Os nomes dos arquivos contêm informações como:
  - `cow_id` (id da vaca)
  - data/hora (`YYYY_MM_DD_HH_MM_SS` ou variações)
  - `cam_ID` (id da câmera)
  - `station_ID` (posição/estação no parlor)

**Campos comuns:**

- `cow_id`: número/ID da vaca
- `YYYY`: ano · `MM`: mês · `DD`: dia
- `HH`: hora · `MM`: minuto · `SS`: segundo
- `cam_ID`: número da câmera
- `station_ID`: posição/estação

> Dica: como existem variações de padrão no nome, mantenha um parser único em `src/data/filename_parser.py`.

---

## Keypoints

Os keypoints representam pontos anatômicos (ex.: **Head, Neck, Withers, Back, Hook, Hip ridge, Tail head, Pin**).  
A lista final (e a **ordem**) deve ser a mesma em:

- ferramenta de anotação
- `data/cows-pose.yaml` (config YOLO)
- scripts de extração de features

<p align="center">
<img src="./docs/keypoints.jpg" alt="Diagrama dos keypoints" width="700" />
</p>

---

## Estrutura do repositório (sugerida)

```text
.
├── docs/
│   └── keypoints.jpg
├── data/
│   ├── raw/                      # imagens originais
│   ├── yolo_pose/                # dataset no formato YOLO Pose
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── labels/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   ├── processed/
│   │   └── features.parquet      # tabela final de features + cow_id
│   └── cows-pose.yaml            # config do dataset para Ultralytics
├── models/
│   ├── yolo_pose/                # runs/checkpoints do YOLO Pose
│   └── classifier/               # modelos do classificador
├── outputs/
│   ├── predictions/              # keypoints previstos (json/csv)
│   ├── metrics/                  # relatórios e métricas
│   └── figures/                  # plots
├── src/
│   ├── data/
│   ├── pose/
│   ├── features/
│   ├── classification/
│   └── utils/
├── notebooks/
├── requirements.txt
└── README.md
```

---

## Setup

### Requisitos

- Python 3.10+
- `ultralytics` (YOLOv8/YOLO11)
- `numpy`, `pandas`, `opencv-python`, `matplotlib`
- `scikit-learn` (classificação)

### Instalação

````bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

---

## Formato YOLO Pose (labels)

Cada imagem tem um arquivo correspondente em `labels/` com linhas no formato:

```text
class x_center y_center width height x1 y1 v1 x2 y2 v2 ... xk yk vk
````

- Coordenadas são **normalizadas** (0–1).
- `v` é a visibilidade (padrão comum):
  - `0` = não rotulado/fora da imagem
  - `1` = rotulado mas ocluído
  - `2` = visível

> Observação: o número de keypoints `k` precisa bater com `kpt_shape` no YAML.

---

## Config do dataset (Ultralytics)

Crie `data/cows-pose.yaml` parecido com isto (ajuste caminhos e keypoints):

```yaml
path: data/yolo_pose
train: images/train
val: images/val
test: images/test

# Uma classe (vaca). Se quiser, pode manter sempre 0.
names:
  0: cow

# Formato: [num_keypoints, dims] (dims=2 -> x,y)
kpt_shape: [8, 2]

# Opcional: nomes dos keypoints (apenas documentação)
keypoints:
  - head
  - neck
  - withers
  - back
  - hook
  - hip_ridge
  - tail_head
  - pin
```

---

## Treinar YOLO Pose

### Treino (CLI)

```bash
yolo pose train model=yolov8n-pose.pt data=data/cows-pose.yaml imgsz=960 epochs=200 batch=16
```

- `imgsz`: ajuste conforme sua resolução (ex.: 640/960/1280)
- `model`: comece com `yolov8n-pose.pt` (rápido) e depois teste `yolov8s-pose.pt`

Os resultados vão para `runs/pose/train*` (por padrão).

### Validação / métricas

```bash
yolo pose val model=runs/pose/train/weights/best.pt data=data/cows-pose.yaml imgsz=960
```

### Inferência (salvar visualizações)

```bash
yolo pose predict model=runs/pose/train/weights/best.pt source=data/yolo_pose/images/test save=True
```

---

## Exportar keypoints para tabela (features)

Fluxo recomendado:

1. Rode inferência no conjunto desejado (train/val/test).
2. Converta as saídas do YOLO (por imagem) em um formato tabular:
   - `image_id`, `cow_id`, `cam_id`, `station_id`, `timestamp`
   - `kp_head_x`, `kp_head_y`, ..., `kp_pin_x`, `kp_pin_y`
3. Gere features:
   - ângulos (neck/withers/back/hip)
   - distâncias e proporções normalizadas
   - estatísticas por vaca (média/desvio/quantis)

Salve em:

- `data/processed/features.parquet` (recomendado) ou `features.csv`

---

## Classificação (identificação da vaca)

Com a tabela de features + `cow_id`:

- Baselines: Logistic Regression, RandomForest, SVM
- Para performance: XGBoost/LightGBM (opcional)

Boas práticas:

- Split por **condição** (ex.: estação/câmera/tempo) e/ou por **vaca**, conforme a pergunta que você quer responder.
- Padronize features (`StandardScaler`) para modelos lineares/SVM.
- Avalie com:
  - Accuracy (top-1)
  - Top-k accuracy (top-3)
  - F1 macro
  - Confusion matrix

---

## Reprodutibilidade

- Fixe seeds (numpy + framework)
- Registre configs por experimento (yaml/json)
- Versione as anotações (quando possível)
- Salve métricas em `outputs/metrics/`

---

1. preparar dataset
2. treinar
3. validar
4. exportar features
5. treinar classificador

## Checklist

- [x] Dataset em `data/yolo_pose/` no formato YOLO Pose
- [x] `data/cows-pose.yaml` configurado com `kpt_shape` correto
- [x] Treino YOLO Pose rodando e salvando `best.pt`
- [x] Export de keypoints para tabela
- [x] Features geradas e analisadas
- [ ] Classificador treinado e avaliado
