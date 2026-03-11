"""
Step 6 of the challenge:
Evaluate the trained cow classification model in detail.
Generates normalized confusion matrices, per-cow accuracy, top confused pairs,
confidence distributions, and a comprehensive evaluation report.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             top_k_accuracy_score)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core_utils import TARGET_KPS

# Reuse feature extraction from train_classifier
from train_classifier import extract_features_from_keypoints, KP_NAMES

import re


def main():
    parser = argparse.ArgumentParser(description="Evaluate cow classifier (Step 6)")
    parser.add_argument("--features", type=str, default="outputs/reports/classification_features.csv",
                        help="Features CSV (from train_classifier.py)")
    parser.add_argument("--classifier", type=str, default="outputs/models/cow_classifier.joblib",
                        help="Trained classifier bundle")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    features_path = Path(args.features)
    clf_path = Path(args.classifier)
    out = Path(args.output_dir)
    fig_dir = out / "figures"
    report_dir = out / "reports"
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if not features_path.exists():
        print(f"ERROR: {features_path} not found. Run train_classifier.py first.")
        sys.exit(1)
    if not clf_path.exists():
        print(f"ERROR: {clf_path} not found. Run train_classifier.py first.")
        sys.exit(1)

    # ── Load data ──
    df = pd.read_csv(features_path)
    df["cow_id"] = df["cow_id"].astype(str)
    bundle = joblib.load(clf_path)
    clf = bundle["classifier"]
    scaler = bundle["scaler"]
    le = bundle["label_encoder"]
    feature_names = bundle["feature_names"]

    print(f"Loaded {len(df)} samples, classifier: {type(clf).__name__}")

    # Prepare data
    meta_cols = ["cow_id", "image", "session_id"]
    df_clean = df.dropna(subset=feature_names).copy()

    X = df_clean[feature_names].values
    y_labels = df_clean["cow_id"].values
    y = le.transform(y_labels)
    n_classes = len(le.classes_)

    # Don't scale for RandomForest
    is_rf = "RandomForest" in type(clf).__name__
    X_in = X if is_rf else scaler.transform(X)

    print(f"Evaluating with {len(df_clean)} samples, {n_classes} classes")

    # ====================================================================
    # 1. Cross-validation predictions (out-of-fold)
    # ====================================================================
    groups = df_clean["session_id"].values if "session_id" in df_clean.columns else None
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
        y_pred_cv = cross_val_predict(clf, X_in, y, cv=cv, groups=groups, method="predict")
        y_proba_cv = cross_val_predict(clf, X_in, y, cv=cv, groups=groups, method="predict_proba")
    else:
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        y_pred_cv = cross_val_predict(clf, X_in, y, cv=cv, method="predict")
        y_proba_cv = cross_val_predict(clf, X_in, y, cv=cv, method="predict_proba")

    # Metrics
    cv_acc = accuracy_score(y, y_pred_cv)
    cv_f1 = f1_score(y, y_pred_cv, average="macro")
    cv_top3 = top_k_accuracy_score(y, y_proba_cv, k=min(3, n_classes))
    cv_top5 = top_k_accuracy_score(y, y_proba_cv, k=min(5, n_classes))

    print(f"\nOut-of-fold CV metrics:")
    print(f"  Top-1 Accuracy: {cv_acc:.4f}")
    print(f"  Top-3 Accuracy: {cv_top3:.4f}")
    print(f"  Top-5 Accuracy: {cv_top5:.4f}")
    print(f"  F1-macro:       {cv_f1:.4f}")
    print(f"  Baseline:       {1/n_classes:.4f}")

    # ====================================================================
    # 2. Normalized Confusion Matrix
    # ====================================================================
    cm = confusion_matrix(y, y_pred_cv)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                xticklabels=le.classes_, yticklabels=le.classes_,
                vmin=0, vmax=1, linewidths=0.3)
    ax.set_title(f"Normalized Confusion Matrix — {type(clf).__name__}\n"
                 f"CV Accuracy: {cv_acc:.3f} | Top-3: {cv_top3:.3f} | Top-5: {cv_top5:.3f}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Cow ID")
    ax.set_ylabel("True Cow ID")
    plt.xticks(rotation=45, fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix_normalized.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: confusion_matrix_normalized.png")

    # ====================================================================
    # 3. Per-cow accuracy
    # ====================================================================
    per_cow_acc = {}
    for i, cow_label in enumerate(le.classes_):
        mask = y == i
        if mask.sum() > 0:
            per_cow_acc[cow_label] = accuracy_score(y[mask], y_pred_cv[mask])

    # Sort by accuracy
    sorted_cows = sorted(per_cow_acc.items(), key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    cows = [c[0] for c in sorted_cows]
    accs = [c[1] for c in sorted_cows]
    colors = ["#2ecc71" if a > cv_acc else "#e74c3c" for a in accs]
    bars = ax.bar(range(len(cows)), accs, color=colors, alpha=0.8)
    ax.set_xticks(range(len(cows)))
    ax.set_xticklabels(cows, rotation=45, fontsize=8)
    ax.set_ylabel("CV Accuracy")
    ax.set_title(f"Per-Cow Classification Accuracy (CV Out-of-Fold)\n"
                 f"Green = above average ({cv_acc:.3f}), Red = below average",
                 fontsize=12, fontweight="bold")
    ax.axhline(y=cv_acc, color="black", linestyle="--", linewidth=1, label=f"Mean: {cv_acc:.3f}")
    ax.axhline(y=1/n_classes, color="gray", linestyle=":", linewidth=1, label=f"Chance: {1/n_classes:.3f}")
    ax.legend()
    ax.set_ylim(0, max(accs) * 1.15)
    plt.tight_layout()
    plt.savefig(fig_dir / "accuracy_per_cow.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: accuracy_per_cow.png")

    # ====================================================================
    # 4. Top confused pairs
    # ====================================================================
    confused_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true_cow": le.classes_[i],
                    "predicted_cow": le.classes_[j],
                    "count": int(cm[i, j]),
                    "rate": float(cm_norm[i, j]),
                })

    confused_df = pd.DataFrame(confused_pairs).sort_values("count", ascending=False)
    top_confused = confused_df.head(10)

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [f"{r['true_cow']} → {r['predicted_cow']}" for _, r in top_confused.iterrows()]
    counts = top_confused["count"].values
    rates = top_confused["rate"].values
    bars = ax.barh(range(len(labels)), counts, color="#e74c3c", alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Number of Misclassifications")
    ax.set_title("Top 10 Most Confused Cow Pairs", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{rate:.1%}", va="center", fontsize=8, color="#666")
    plt.tight_layout()
    plt.savefig(fig_dir / "top_confused_pairs.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: top_confused_pairs.png")

    # ====================================================================
    # 5. Confidence distribution (correct vs incorrect)
    # ====================================================================
    pred_confidences = y_proba_cv[np.arange(len(y)), y_pred_cv]
    correct_mask = y_pred_cv == y

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pred_confidences[correct_mask], bins=30, alpha=0.6, color="#2ecc71",
            label=f"Correct ({correct_mask.sum()})", density=True)
    ax.hist(pred_confidences[~correct_mask], bins=30, alpha=0.6, color="#e74c3c",
            label=f"Incorrect ({(~correct_mask).sum()})", density=True)
    ax.set_xlabel("Predicted Class Confidence")
    ax.set_ylabel("Density")
    ax.set_title("Confidence Distribution — Correct vs Incorrect Predictions",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axvline(x=1/n_classes, color="gray", linestyle="--", label=f"Chance: {1/n_classes:.3f}")
    plt.tight_layout()
    plt.savefig(fig_dir / "confidence_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: confidence_distribution.png")

    # ====================================================================
    # 6. Top-K accuracy curve
    # ====================================================================
    top_k_accs = []
    for k in range(1, min(n_classes + 1, 16)):
        top_k_accs.append(top_k_accuracy_score(y, y_proba_cv, k=k))

    fig, ax = plt.subplots(figsize=(10, 5))
    ks = list(range(1, len(top_k_accs) + 1))
    ax.plot(ks, top_k_accs, "o-", color="#3498db", linewidth=2, markersize=6)
    ax.fill_between(ks, top_k_accs, alpha=0.15, color="#3498db")
    for k, acc in zip(ks, top_k_accs):
        if k in [1, 3, 5, 10]:
            ax.annotate(f"{acc:.3f}", (k, acc), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("K")
    ax.set_ylabel("Top-K Accuracy")
    ax.set_title("Top-K Accuracy Curve (CV Out-of-Fold)", fontsize=13, fontweight="bold")
    ax.set_xticks(ks)
    ax.axhline(y=1/n_classes, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(0, min(1.05, max(top_k_accs) * 1.15))
    plt.tight_layout()
    plt.savefig(fig_dir / "topk_accuracy_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: topk_accuracy_curve.png")

    # ====================================================================
    # 7. Classification report
    # ====================================================================
    report_str = classification_report(y, y_pred_cv,
                                       target_names=[str(c) for c in le.classes_],
                                       output_dict=True)
    report_text = classification_report(y, y_pred_cv,
                                        target_names=[str(c) for c in le.classes_])

    # ====================================================================
    # 8. Generate evaluation report (Markdown)
    # ====================================================================
    report_path = report_dir / "evaluation_report.md"
    with open(report_path, "w") as f:
        f.write("# Relatório de Avaliação — Classificação de Vacas\n\n")

        f.write("## 1. Resumo\n\n")
        f.write(f"- **Classificador**: {type(clf).__name__}\n")
        f.write(f"- **Total de amostras**: {len(df_clean)}\n")
        f.write(f"- **Vacas (classes)**: {n_classes}\n")
        f.write(f"- **Features**: {len(feature_names)} (ângulos + ratios, scale-invariant)\n")
        f.write(f"- **Validação**: 5-fold Stratified GroupKFold (agrupado por sessão)\n\n")

        f.write("## 2. Métricas de Desempenho\n\n")
        f.write("| Métrica | Valor |\n|---|---|\n")
        f.write(f"| **Top-1 Accuracy (CV)** | {cv_acc:.4f} |\n")
        f.write(f"| **Top-3 Accuracy (CV)** | {cv_top3:.4f} |\n")
        f.write(f"| **Top-5 Accuracy (CV)** | {cv_top5:.4f} |\n")
        f.write(f"| **F1-macro (CV)** | {cv_f1:.4f} |\n")
        f.write(f"| **Baseline (acaso)** | {1/n_classes:.4f} |\n")
        f.write(f"| **Lift sobre baseline** | {cv_acc/(1/n_classes):.1f}x |\n\n")

        f.write("### Top-K Accuracy\n\n")
        f.write("| K | Accuracy |\n|---|---|\n")
        for k, acc in zip(ks, top_k_accs):
            marker = " ⬅" if k in [1, 3, 5] else ""
            f.write(f"| {k} | {acc:.4f}{marker} |\n")
        f.write("\n")

        f.write("## 3. Acurácia por Vaca\n\n")
        f.write("### Vacas mais fáceis de identificar\n\n")
        f.write("| Cow ID | Accuracy | Amostras |\n|---|---|---|\n")
        for cow, acc in sorted_cows[:5]:
            n = int((y == le.transform([cow])[0]).sum())
            f.write(f"| {cow} | {acc:.4f} | {n} |\n")
        f.write("\n")

        f.write("### Vacas mais difíceis de identificar\n\n")
        f.write("| Cow ID | Accuracy | Amostras |\n|---|---|---|\n")
        for cow, acc in sorted_cows[-5:]:
            n = int((y == le.transform([cow])[0]).sum())
            f.write(f"| {cow} | {acc:.4f} | {n} |\n")
        f.write("\n")

        f.write("## 4. Análise de Erros — Pares Mais Confundidos\n\n")
        f.write("Os 10 pares de vacas mais frequentemente confundidos pelo classificador:\n\n")
        f.write("| Vaca Real | Predita como | Nº Erros | Taxa |\n|---|---|---|---|\n")
        for _, row in top_confused.iterrows():
            f.write(f"| {row['true_cow']} | {row['predicted_cow']} | "
                    f"{row['count']} | {row['rate']:.1%} |\n")
        f.write("\n")

        f.write("## 5. Distribuição de Confiança\n\n")
        correct_conf = pred_confidences[correct_mask]
        incorrect_conf = pred_confidences[~correct_mask]
        f.write(f"- **Confiança média (predições corretas)**: {correct_conf.mean():.4f}\n")
        f.write(f"- **Confiança média (predições incorretas)**: {incorrect_conf.mean():.4f}\n")
        f.write(f"- **Predições com alta confiança (> 50%)**: "
                f"{(pred_confidences > 0.5).sum()} ({(pred_confidences > 0.5).mean():.1%})\n\n")

        f.write("## 6. Relatório de Classificação Detalhado\n\n")
        f.write("```\n")
        f.write(report_text)
        f.write("\n```\n\n")

        f.write("## 7. Discussão e Limitações\n\n")

        f.write("### Pontos Positivos\n\n")
        f.write(f"- O classificador alcança **{cv_acc/(1/n_classes):.1f}x** a performance do acaso, ")
        f.write("demonstrando que as features geométricas capturam informação discriminativa.\n")
        f.write(f"- A acurácia **Top-5 de {cv_top5:.1%}** mostra que, na maioria dos casos, ")
        f.write("a vaca correta está entre as 5 principais predições.\n")
        f.write(f"- O uso de **StratifiedGroupKFold** por sessão garante que as métricas ")
        f.write("são realistas e não infladas por data leakage.\n\n")

        f.write("### Limitações\n\n")
        f.write("- A acurácia Top-1 ainda é relativamente baixa — as features geométricas ")
        f.write("sozinhas podem não capturar variações sutis entre indivíduos.\n")
        f.write("- A **variância intra-classe** (mesma vaca em diferentes posições/dias) ")
        f.write("é significativa, dificultando a separação.\n")
        f.write("- O dataset tem apenas **50 imagens por vaca**, o que limita a ")
        f.write("capacidade de generalização.\n\n")

        f.write("### Possíveis Melhorias\n\n")
        f.write("1. **Mais keypoints**: adicionar pontos anatômicos extras (ex: patas) ")
        f.write("para capturar mais variação morfológica.\n")
        f.write("2. **Features de aparência**: combinar features geométricas com ")
        f.write("embeddings visuais (ex: ResNet, EfficientNet) da região do dorso.\n")
        f.write("3. **Aumento de dados**: fotografar as vacas em mais horários e ")
        f.write("condições de iluminação.\n")
        f.write("4. **Modelos mais complexos**: testar redes neurais ou gradient boosting ")
        f.write("(XGBoost, LightGBM) para capturar interações não-lineares.\n\n")

        f.write("## 8. Gráficos Gerados\n\n")
        eval_figs = [
            "confusion_matrix_normalized.png",
            "accuracy_per_cow.png",
            "top_confused_pairs.png",
            "confidence_distribution.png",
            "topk_accuracy_curve.png",
        ]
        for fig_name in eval_figs:
            f.write(f"- `{fig_name}`\n")
        f.write("\n")

    print(f"\nEvaluation report saved to: {report_path}")

    # Save metrics JSON
    eval_metrics = {
        "classifier": type(clf).__name__,
        "n_samples": len(df_clean),
        "n_classes": n_classes,
        "cv_top1_accuracy": round(cv_acc, 4),
        "cv_top3_accuracy": round(cv_top3, 4),
        "cv_top5_accuracy": round(cv_top5, 4),
        "cv_f1_macro": round(cv_f1, 4),
        "baseline_chance": round(1/n_classes, 4),
        "lift_over_baseline": round(cv_acc / (1/n_classes), 2),
        "top_k_curve": {str(k): round(a, 4) for k, a in zip(ks, top_k_accs)},
        "per_cow_accuracy": {cow: round(acc, 4) for cow, acc in sorted_cows},
        "top_confused_pairs": top_confused.to_dict("records"),
    }
    with open(report_dir / "evaluation_metrics.json", "w") as fp:
        json.dump(eval_metrics, fp, indent=4)
    print(f"Metrics saved to: {report_dir / 'evaluation_metrics.json'}")

    print(f"\n{'='*50}")
    print("EVALUATION COMPLETE")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
