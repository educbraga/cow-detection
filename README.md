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
