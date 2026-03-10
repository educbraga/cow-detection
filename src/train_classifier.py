"""
Steps 5 & 6 of the challenge:
- Extract geometric features from the classification dataset (with cow_id)
- Train classifiers (LogisticRegression, RandomForest, SVM)
- Evaluate with accuracy, top-3 accuracy, F1-macro, confusion matrix
"""
import argparse
import json
import joblib
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, top_k_accuracy_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core_utils import TARGET_KPS

# ───────── Feature extraction (same logic as extract_features.py) ─────────

KP_NAMES = TARGET_KPS

def angle_between(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0)))

def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def extract_features_from_keypoints(kps_dict):
    features = {}
    for name in KP_NAMES:
        features[f"kp_{name.replace(' ','_')}_x"] = kps_dict.get(name, (np.nan, np.nan))[0]
        features[f"kp_{name.replace(' ','_')}_y"] = kps_dict.get(name, (np.nan, np.nan))[1]

    angle_defs = [
        ("angle_withers_back_hip",        ["withers", "back", "hip"]),
        ("angle_back_hip_tail",           ["back", "hip", "tail head"]),
        ("angle_hip_tail_pin",            ["hip", "tail head", "pin up"]),
        ("angle_hook_up_hip_hook_down",   ["hook up", "hip", "hook down"]),
        ("angle_pin_up_tail_pin_down",    ["pin up", "tail head", "pin down"]),
    ]
    for fname, (a, b, c) in angle_defs:
        if all(k in kps_dict for k in [a, b, c]):
            features[fname] = angle_between(kps_dict[a], kps_dict[b], kps_dict[c])
        else:
            features[fname] = np.nan

    dist_pairs = [
        ("withers","back"), ("back","hip"), ("hip","tail head"),
        ("tail head","pin up"), ("tail head","pin down"),
        ("hook up","hook down"), ("pin up","pin down"),
        ("withers","tail head"), ("withers","hip"),
    ]
    for a, b in dist_pairs:
        fn = f"dist_{a.replace(' ','_')}_{b.replace(' ','_')}"
        features[fn] = distance(kps_dict[a], kps_dict[b]) if (a in kps_dict and b in kps_dict) else np.nan

    spine = features.get("dist_withers_tail_head", np.nan)
    if not np.isnan(spine) and spine > 1e-6:
        for k in list(features.keys()):
            if k.startswith("dist_") and k != "dist_withers_tail_head":
                features[f"ratio_{k[5:]}"] = features[k] / spine
    hook_w = features.get("dist_hook_up_hook_down", np.nan)
    pin_w  = features.get("dist_pin_up_pin_down", np.nan)
    if not np.isnan(hook_w) and not np.isnan(pin_w) and pin_w > 1e-6:
        features["ratio_hook_to_pin_width"] = hook_w / pin_w
    else:
        features["ratio_hook_to_pin_width"] = np.nan
    return features

# ───────── Main pipeline ─────────

def main():
    parser = argparse.ArgumentParser(description="Train cow classifiers (steps 5-6)")
    parser.add_argument("--model", type=str, default="outputs/models/best_pose.pt")
    parser.add_argument("--dataset", type=str, default="data/dataset_classificação")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    model_path = Path(args.model)
    dataset_dir = Path(args.dataset)
    out = Path(args.output_dir)
    fig_dir = out / "figures"
    report_dir = out / "reports"
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}"); sys.exit(1)
    if not dataset_dir.exists():
        print(f"ERROR: dataset not found at {dataset_dir}"); sys.exit(1)

    model = YOLO(str(model_path))

    # ── 1. Build feature table with cow_id ──
    cow_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    print(f"Found {len(cow_dirs)} cow classes")

    rows = []
    for cow_dir in cow_dirs:
        cow_id = cow_dir.name
        imgs = sorted(list(cow_dir.glob("*.jpg")) + list(cow_dir.glob("*.jpeg")))
        for img_path in imgs:
            try:
                results = model(str(img_path), conf=args.conf, verbose=False)
                if not results or not results[0].keypoints or len(results[0].keypoints) == 0:
                    continue
                r = results[0]
                best = r.boxes.conf.argmax().item() if r.boxes is not None and len(r.boxes) > 0 else 0
                kps_data = r.keypoints.data[best].cpu().numpy()
                kps_dict = {}
                for i, name in enumerate(KP_NAMES):
                    if i < len(kps_data):
                        x, y, c = kps_data[i]
                        if c > 0.3:
                            kps_dict[name] = (float(x), float(y))
                if len(kps_dict) < 3:
                    continue
                feats = extract_features_from_keypoints(kps_dict)
                feats["cow_id"] = cow_id
                feats["image"] = img_path.name
                rows.append(feats)
            except Exception as e:
                print(f"  Error {img_path.name}: {e}")
        print(f"  cow {cow_id}: {sum(1 for r in rows if r['cow_id']==cow_id)} features extracted")

    df = pd.DataFrame(rows)
    df.to_csv(out / "reports" / "classification_features.csv", index=False)
    print(f"\nTotal samples: {len(df)} (from {len(cow_dirs)} cows)")

    # ── 2. Prepare X, y ──
    meta_cols = ["cow_id", "image"]
    kp_raw_cols = [c for c in df.columns if c.startswith("kp_")]  # raw coords are scale-dependent
    feature_cols = [c for c in df.columns if c not in meta_cols + kp_raw_cols
                    and df[c].dtype in ["float64", "float32", "int64"]]
    
    # Use angles + ratios (scale-invariant) as primary features
    angle_cols = [c for c in feature_cols if c.startswith("angle_")]
    ratio_cols = [c for c in feature_cols if c.startswith("ratio_")]
    selected = angle_cols + ratio_cols
    print(f"Using {len(selected)} features: {len(angle_cols)} angles + {len(ratio_cols)} ratios")

    df_clean = df.dropna(subset=selected).copy()
    print(f"Samples after dropping NaN: {len(df_clean)} ({len(df)-len(df_clean)} dropped)")

    if len(df_clean) < 30:
        print("ERROR: Not enough samples. Aborting."); sys.exit(1)

    X = df_clean[selected].values
    le = LabelEncoder()
    y = le.fit_transform(df_clean["cow_id"].values)
    n_classes = len(le.classes_)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 3. Train & evaluate classifiers ──
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=args.seed),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=-1),
        "SVM_RBF": SVC(kernel="rbf", probability=True, random_state=args.seed),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    results = {}

    for name, clf in classifiers.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")

        X_in = X_scaled if name != "RandomForest" else X  # RF doesn't need scaling

        # Cross-validation for accuracy
        cv_scores = cross_val_score(clf, X_in, y, cv=cv, scoring="accuracy")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Full fit for final metrics & confusion matrix
        clf.fit(X_in, y)
        y_pred = clf.predict(X_in)
        y_proba = clf.predict_proba(X_in) if hasattr(clf, "predict_proba") else None

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="macro")
        top3 = top_k_accuracy_score(y, y_proba, k=min(3, n_classes)) if y_proba is not None else "N/A"

        print(f"  Train Accuracy: {acc:.4f}")
        print(f"  Train F1-macro: {f1:.4f}")
        print(f"  Train Top-3 Acc: {top3:.4f}" if isinstance(top3, float) else f"  Train Top-3 Acc: {top3}")

        results[name] = {
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4),
            "train_accuracy": round(float(acc), 4),
            "train_f1_macro": round(float(f1), 4),
            "train_top3_accuracy": round(float(top3), 4) if isinstance(top3, float) else top3,
        }

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=le.classes_, yticklabels=le.classes_)
        ax.set_title(f"Confusion Matrix — {name}\nCV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted cow_id")
        ax.set_ylabel("True cow_id")
        plt.xticks(rotation=45, fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()
        plt.savefig(fig_dir / f"confusion_matrix_{name}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Classification report
        report_str = classification_report(y, y_pred, target_names=[str(c) for c in le.classes_])
        print(f"\n{report_str}")

    # ── 4. Feature importance (RandomForest) ──
    rf = classifiers["RandomForest"]
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(selected)), importances[idx[::-1]], color="#4C72B0")
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels([selected[i].replace("_"," ").title() for i in idx[::-1]], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (RandomForest)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_importance_rf.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 5. Comparison bar chart ──
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(results.keys())
    cv_accs = [results[n]["cv_accuracy_mean"] for n in names]
    cv_stds = [results[n]["cv_accuracy_std"] for n in names]
    bars = ax.bar(names, cv_accs, yerr=cv_stds, capsize=5, color=["#4C72B0", "#55A868", "#C44E52"], alpha=0.85)
    for bar, val in zip(bars, cv_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("CV Accuracy (5-fold)")
    ax.set_title("Classifier Comparison — Cow Identification", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1/n_classes, color="gray", linestyle="--", label=f"Chance ({1/n_classes:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "classifier_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 6. Save final report ──
    report = {
        "n_samples": len(df_clean),
        "n_classes": n_classes,
        "n_features": len(selected),
        "features_used": selected,
        "results": results,
    }
    with open(report_dir / "classification_results.json", "w") as f:
        json.dump(report, f, indent=4)

    # Markdown summary
    with open(report_dir / "classification_summary.md", "w") as f:
        f.write("# Classification Results — Cow Identification\n\n")
        f.write(f"- **Samples**: {len(df_clean)} (from {n_classes} cows)\n")
        f.write(f"- **Features**: {len(selected)} (angles + ratios, scale-invariant)\n")
        f.write(f"- **Evaluation**: 5-fold Stratified Cross-Validation\n\n")
        f.write("## Results\n\n")
        f.write("| Classifier | CV Accuracy | ± Std | Train F1-macro | Train Top-3 |\n")
        f.write("|---|---|---|---|---|\n")
        for n, r in results.items():
            f.write(f"| {n} | {r['cv_accuracy_mean']:.4f} | {r['cv_accuracy_std']:.4f} | {r['train_f1_macro']:.4f} | {r['train_top3_accuracy']:.4f} |\n")
        f.write(f"\n> Baseline (chance): {1/n_classes:.4f}\n")

    # ── 7. Save trained models for inference ──
    models_dir = out / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    classifier_bundle = {
        "classifier": classifiers["RandomForest"],
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": selected,
    }
    bundle_path = models_dir / "cow_classifier.joblib"
    joblib.dump(classifier_bundle, bundle_path)
    print(f"\nClassifier saved to: {bundle_path}")

    print(f"\n{'='*50}")
    print("CLASSIFICATION COMPLETE")
    print(f"{'='*50}")
    print(f"Classifier: {bundle_path}")
    print(f"Report: {report_dir / 'classification_results.json'}")
    print(f"Summary: {report_dir / 'classification_summary.md'}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
