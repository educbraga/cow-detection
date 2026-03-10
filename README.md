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

## 🚀 Como Executar o Projeto

Se você acabou de clonar este repositório e quer rodar o projeto na sua máquina, siga o passo a passo abaixo. Todos os comandos devem ser executados no terminal, dentro da pasta raiz do projeto (`cow-detection`).

### 1. Preparando o Ambiente

**Criar e ativar o ambiente virtual (Python):**
Recomendamos o uso do `venv` para manter as bibliotecas isoladas.

```bash
# Cria o ambiente virtual na pasta .venv
python -m venv .venv

# Ativa o ambiente virtual (Mac/Linux)
source .venv/bin/activate
# Se estiver no Windows, use: .venv\Scripts\activate
```

**Instalar as dependências:**
Com o ambiente ativado, instale as bibliotecas necessárias (como `ultralytics`, `pandas`, `scikit-learn`, etc).

```bash
pip install -r requirements.txt
```

---

### 2. A Ordem dos Scripts (`src/`)

A pasta `src/` contém os scripts principais do pipeline, desde a preparação dos dados até a extração de métricas. Eles foram desenhados para serem executados em uma sequência lógica.

Aqui está a ordem sugerida e para que cada um serve:

| Ordem  | Script                                           | O que ele faz?                                                                                                                                                                                                                                              |
| :----: | :----------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1º** | `validate_annotations.py` / `inspect_dataset.py` | **Inspeção de Dados:** Ler as anotações geradas (Label Studio, etc.), verificar se existem _keypoints_ ausentes ou duplicados nas imagens e garantir que o dataset está saudável.                                                                           |
| **2º** | `convert_to_yolo_pose.py`                        | **Conversão:** Pega as anotações limpas e as converte para o formato de texto que o `YOLO Pose` exige (coordenadas das tags, classes e visibilidade dos pontos).                                                                                            |
| **3º** | `make_subset.py`                                 | **Divisão de Dados:** Separa o dataset limpo em um _subconjunto_ ou em pastas finais de treinamento (`train`) e validação (`val`).                                                                                                                          |
| **4º** | `train_pose.py`                                  | **Treinamento (Modelo de IA):** Inicia o treinamento do modelo YOLO11-Pose para aprender a identificar os pontos anatômicos das vacas na imagem. Salva o melhor modelo em `.pt`.                                                                            |
| **5º** | `evaluate_pose.py` / `validate.py`               | **Avaliação da IA:** Pega o modelo que você acabou de treinar e avalia nas imagens de teste para calcular as métricas de performance (mAP, Precisão, Recall).                                                                                               |
| **6º** | `extract_features.py`                            | **Extração de Características:** Roda o modelo treinado em todas as imagens para capturar as coordenadas e, com elas, calcula as **features geométricas** (ângulos, distâncias e proporções). Salva o resultado em `features.csv`.                          |
| **7º** | `analyze_features.py`                            | **Análise Científica:** Lê o arquivo `features.csv` gerado e cria os gráficos estatísticos (mapas de calor de correlação e histogramas). O objetivo é descobrir quais medidas (ex: ângulo do quadril) servem como "impressão digital" geométrica do animal. |

> **💡 Scripts Extras e Auxiliares:**
>
> - `core_utils.py`: **Não deve ser executado diretamente**. Contém funções vitais que são importadas pelos outros scripts (como parseamento dos nomes bizarros dos arquivos, cálculos matemáticos de distância e ângulo).
> - `visualize_predictions.py` / `debug_visualize.py`: Scripts visuais! Eles desenham os pontinhos e as linhas de "esqueleto" em cima da foto da vaca. Muito úteis se você quiser "enxergar" o que a rede neural está prevendo no visual.
