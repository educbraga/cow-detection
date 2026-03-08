# Cow Detection Pipeline — Walkthrough

## Resumo

Pipeline de detecção de keypoints em vacas (vista superior) executado com sucesso. Etapas 1-5 do challenge completas; etapa 6 (classificador) pendente de `cow_id`.

## Etapa 0 — Preparação

- Recriação do `.venv` e instalação de dependências (ultralytics, pandas, scikit-learn, seaborn)
- [train_pose.py](file:///Users/eduardobraga/www/ai/cow-detection/src/train_pose.py) → `device="mps"` (Apple M4)
- [core_utils.py](file:///Users/eduardobraga/www/ai/cow-detection/src/core_utils.py) → parser expandido para 3 padrões de filename

## Etapa 1 — Dataset YOLO Pose

- **1029 anotações** processadas, **117 rejeitadas** (keypoints duplicados/ausentes)
- **912 válidas** → subset de **150 imagens** (120 train / 30 val)
- Output: `data/subset_yolo_pose/` com `data.yaml` (8 keypoints, 3 dims)

## Etapa 2 — Treino YOLO Pose

Modelo: `yolo11n-pose.pt` — 100 épocas em **16.5 min** no MPS (Apple M4)

### Métricas Finais

| Métrica | Bounding Box | Keypoint Pose |
|---|---|---|
| **mAP50** | 0.995 | 0.995 |
| **mAP50-95** | 0.933 | 0.876 |
| **Precision** | 0.998 | 0.998 |
| **Recall** | 1.000 | 1.000 |
| **F1-Score** | 0.999 | 0.999 |

## Etapa 3 — Avaliação

Validação confirmada no test set: mAP50=0.995, mAP50-95=0.940.
Relatório em [metrics.json](file:///Users/eduardobraga/www/ai/cow-detection/outputs/reports/metrics.json).

## Etapa 4 — Extração de Features

Inferência em **1029 imagens** → [features.csv](file:///Users/eduardobraga/www/ai/cow-detection/data/processed/features.csv) (1029 rows, 39 colunas):
- **16 coordenadas** raw de keypoints (x, y)
- **5 ângulos**: withers-back-hip, back-hip-tail, hip-tail-pin, hook_up-hip-hook_down, pin_up-tail-pin_down
- **9 distâncias**: entre pares de keypoints
- **9 proporções**: normalizadas pelo comprimento da spine (withers→tail_head)

## Etapa 5 — Análise Descritiva

### Distribuição dos Ângulos
![Histograms of angles](/Users/eduardobraga/.gemini/antigravity/brain/14b24cd8-2394-444b-9ad5-d0646cea9ace/histograms_angles.png)

### Correlação entre Features
![Correlation heatmap](/Users/eduardobraga/.gemini/antigravity/brain/14b24cd8-2394-444b-9ad5-d0646cea9ace/correlation_heatmap.png)

### Pairplot dos Ângulos
![Pairplot angles](/Users/eduardobraga/.gemini/antigravity/brain/14b24cd8-2394-444b-9ad5-d0646cea9ace/pairplot_angles.png)

### Variação por Estação
![Boxplot hip angle by station](/Users/eduardobraga/.gemini/antigravity/brain/14b24cd8-2394-444b-9ad5-d0646cea9ace/boxplot_angle_hip_tail_pin_by_station.png)

### Features Mais Discriminativas (maior CV)

| Feature | CV (%) | Potencial p/ Identificação |
|---|---|---|
| `ratio_tail_head_pin_up` | 36.6% | ⭐ Alto |
| `ratio_hip_tail_head` | 15.2% | ⭐ Alto |
| `ratio_tail_head_pin_down` | 15.0% | ⭐ Alto |
| `angle_pin_up_tail_pin_down` | 12.6% | ⭐ Médio-alto |
| `ratio_withers_back` | 10.1% | Médio |

## Arquivos Gerados

| Arquivo | Descrição |
|---|---|
| [best_pose.pt](file:///Users/eduardobraga/www/ai/cow-detection/outputs/models/best_pose.pt) | Modelo treinado (5.7MB) |
| [features.csv](file:///Users/eduardobraga/www/ai/cow-detection/data/processed/features.csv) | 1029 amostras × 39 features |
| [feature_analysis.md](file:///Users/eduardobraga/www/ai/cow-detection/outputs/reports/feature_analysis.md) | Relatório descritivo |
| `outputs/figures/` | 7 gráficos de análise |
| [extract_features.py](file:///Users/eduardobraga/www/ai/cow-detection/src/extract_features.py) | Script de extração |
| [analyze_features.py](file:///Users/eduardobraga/www/ai/cow-detection/src/analyze_features.py) | Script de análise |

## Pendências

- **Etapa 6 — Classificador**: Requer mapeamento `cow_id → imagem`. Quando disponível, rodar `train_classifier.py` (a ser criado).
