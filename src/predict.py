"""
Identify which cow is in a given image.

Usage:
    python3 src/predict.py --image path/to/image.jpg

Requires:
    - outputs/models/best_pose.pt   (YOLO Pose model)
    - outputs/models/cow_classifier.joblib (trained classifier)
"""
import argparse
import math
import sys
from pathlib import Path

import joblib
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core_utils import TARGET_KPS

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


def main():
    parser = argparse.ArgumentParser(description="Identify which cow is in an image")
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument("--pose-model", type=str, default="outputs/models/best_pose.pt",
                        help="Path to the YOLO Pose model")
    parser.add_argument("--classifier", type=str, default="outputs/models/cow_classifier.joblib",
                        help="Path to the trained classifier bundle")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to show")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    args = parser.parse_args()

    # ── Validate inputs ──
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"❌ Image not found: {img_path}")
        sys.exit(1)

    pose_path = Path(args.pose_model)
    if not pose_path.exists():
        print(f"❌ YOLO Pose model not found: {pose_path}")
        print("   Run: python3 src/train_pose.py")
        sys.exit(1)

    clf_path = Path(args.classifier)
    if not clf_path.exists():
        print(f"❌ Classifier not found: {clf_path}")
        print("   Run: python3 src/train_classifier.py")
        sys.exit(1)

    # ── Load models ──
    print(f"📷 Image: {img_path.name}")
    yolo = YOLO(str(pose_path))
    bundle = joblib.load(clf_path)
    clf = bundle["classifier"]
    scaler = bundle["scaler"]
    le = bundle["label_encoder"]
    feature_names = bundle["feature_names"]

    # ── Run YOLO Pose inference ──
    results = yolo(str(img_path), conf=args.conf, verbose=False)

    if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
        print("❌ No cow detected in the image.")
        sys.exit(1)

    r = results[0]
    n_detections = len(r.boxes) if r.boxes is not None else 0
    print(f"🐄 Detected {n_detections} cow(s)")

    best = r.boxes.conf.argmax().item() if r.boxes is not None and len(r.boxes) > 0 else 0
    kps_data = r.keypoints.data[best].cpu().numpy()

    kps_dict = {}
    for i, name in enumerate(KP_NAMES):
        if i < len(kps_data):
            x, y, c = kps_data[i]
            if c > 0.3:
                kps_dict[name] = (float(x), float(y))

    if len(kps_dict) < 3:
        print(f"❌ Only {len(kps_dict)} keypoints detected (minimum 3 needed).")
        sys.exit(1)

    print(f"📍 Keypoints detected: {len(kps_dict)}/8")

    # ── Extract features ──
    features = extract_features_from_keypoints(kps_dict)

    X = np.array([[features.get(f, np.nan) for f in feature_names]])

    if np.any(np.isnan(X)):
        nan_feats = [f for f, v in zip(feature_names, X[0]) if np.isnan(v)]
        print(f"⚠️  Missing features: {nan_feats}")
        # Fill NaN with 0 for prediction (imprecise but allows a prediction)
        X = np.nan_to_num(X, nan=0.0)

    # ── Predict ──
    proba = clf.predict_proba(X)[0]
    top_indices = np.argsort(proba)[::-1][:args.top_k]

    print(f"\n{'='*40}")
    print(f"🏆 PREDICTION — Top {args.top_k}")
    print(f"{'='*40}")
    for rank, idx in enumerate(top_indices, 1):
        cow_id = le.classes_[idx]
        confidence = proba[idx] * 100
        bar = "█" * int(confidence / 2) + "░" * (50 - int(confidence / 2))
        print(f"  {rank}. Cow {cow_id}  {bar}  {confidence:.1f}%")

    best_cow = le.classes_[top_indices[0]]
    best_conf = proba[top_indices[0]] * 100
    print(f"\n→ Best guess: Cow **{best_cow}** ({best_conf:.1f}% confidence)")


if __name__ == "__main__":
    main()
