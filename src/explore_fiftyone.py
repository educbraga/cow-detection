"""
FiftyOne interactive visualization of the cow detection dataset.
Loads the classification dataset, runs YOLO Pose inference, and displays
results in an interactive dashboard.

Usage:
    python3 src/explore_fiftyone.py

Opens a FiftyOne dashboard in the browser at http://localhost:5151
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import fiftyone as fo
from ultralytics import YOLO
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core_utils import TARGET_KPS
from train_classifier import extract_features_from_keypoints, KP_NAMES


def main():
    parser = argparse.ArgumentParser(description="FiftyOne dataset explorer")
    parser.add_argument("--dataset", type=str, default="data/dataset_classificação",
                        help="Path to classification dataset")
    parser.add_argument("--pose-model", type=str, default="outputs/models/best_pose.pt",
                        help="Path to YOLO Pose model")
    parser.add_argument("--classifier", type=str, default="outputs/models/cow_classifier.joblib",
                        help="Path to trained classifier bundle")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO confidence threshold")
    parser.add_argument("--max-per-cow", type=int, default=10,
                        help="Max images per cow to load (for speed)")
    parser.add_argument("--name", type=str, default="cow-detection",
                        help="FiftyOne dataset name")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    pose_path = Path(args.pose_model)
    clf_path = Path(args.classifier)

    if not dataset_dir.exists():
        print(f"ERROR: Dataset not found at {dataset_dir}"); sys.exit(1)
    if not pose_path.exists():
        print(f"ERROR: YOLO model not found at {pose_path}"); sys.exit(1)
    if not clf_path.exists():
        print(f"ERROR: Classifier not found at {clf_path}"); sys.exit(1)

    # Load models
    yolo = YOLO(str(pose_path))
    bundle = joblib.load(clf_path)
    clf = bundle["classifier"]
    scaler = bundle["scaler"]
    le = bundle["label_encoder"]
    feature_names = bundle["feature_names"]

    # Delete existing dataset with same name
    if fo.dataset_exists(args.name):
        fo.delete_dataset(args.name)

    dataset = fo.Dataset(args.name, persistent=True)
    dataset.info = {
        "description": "Cow identification dataset with YOLO Pose keypoints and classifier predictions",
        "n_classes": len(le.classes_),
    }

    cow_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    print(f"Found {len(cow_dirs)} cow directories")

    total = 0
    correct = 0

    for cow_dir in cow_dirs:
        cow_id = cow_dir.name
        imgs = sorted(list(cow_dir.glob("*.jpg")) + list(cow_dir.glob("*.jpeg")))

        if args.max_per_cow > 0:
            imgs = imgs[:args.max_per_cow]

        for img_path in imgs:
            try:
                # Run YOLO Pose inference
                results = yolo(str(img_path), conf=args.conf, verbose=False)
                if not results or not results[0].keypoints or len(results[0].keypoints) == 0:
                    continue

                r = results[0]
                best = r.boxes.conf.argmax().item() if r.boxes is not None and len(r.boxes) > 0 else 0
                det_conf = float(r.boxes.conf[best].cpu())

                # Get bounding box (normalized for FiftyOne)
                if r.boxes is not None and len(r.boxes) > 0:
                    box = r.boxes.xyxyn[best].cpu().numpy()
                    x1, y1, x2, y2 = box
                    # FiftyOne format: [x, y, width, height] normalized
                    fo_box = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                else:
                    fo_box = None

                # Get keypoints
                kps_data = r.keypoints.data[best].cpu().numpy()
                kps_dict = {}
                fo_keypoints = []
                img_h, img_w = r.orig_shape

                for i, name in enumerate(KP_NAMES):
                    if i < len(kps_data):
                        x, y, c = kps_data[i]
                        if c > 0.3:
                            kps_dict[name] = (float(x), float(y))
                            # Normalize for FiftyOne
                            fo_keypoints.append((float(x / img_w), float(y / img_h)))
                        else:
                            fo_keypoints.append((0, 0))
                    else:
                        fo_keypoints.append((0, 0))

                if len(kps_dict) < 3:
                    continue

                # Extract features and predict
                features = extract_features_from_keypoints(kps_dict)
                X = np.array([[features.get(f, np.nan) for f in feature_names]])
                X = np.nan_to_num(X, nan=0.0)

                is_rf = "RandomForest" in type(clf).__name__
                X_in = X if is_rf else scaler.transform(X)

                proba = clf.predict_proba(X_in)[0]
                pred_idx = np.argmax(proba)
                pred_cow = le.classes_[pred_idx]
                pred_conf = float(proba[pred_idx])

                is_correct = str(pred_cow) == str(cow_id)
                total += 1
                if is_correct:
                    correct += 1

                # Create FiftyOne sample
                sample = fo.Sample(filepath=str(img_path.resolve()))
                sample["ground_truth"] = fo.Classification(label=str(cow_id))
                sample["prediction"] = fo.Classification(
                    label=str(pred_cow),
                    confidence=pred_conf,
                )
                sample["correct"] = is_correct
                sample["det_confidence"] = det_conf
                sample["n_keypoints"] = len(kps_dict)

                # Add detection with keypoints
                if fo_box is not None:
                    kp_points = [fo.Keypoint(
                        label="cow",
                        points=fo_keypoints,
                    )]
                    sample["keypoints"] = fo.Keypoints(keypoints=kp_points)

                    sample["detection"] = fo.Detections(detections=[
                        fo.Detection(
                            label=str(cow_id),
                            bounding_box=fo_box,
                            confidence=det_conf,
                        )
                    ])

                # Top-3 predictions
                top3_idx = np.argsort(proba)[::-1][:3]
                top3_str = ", ".join([
                    f"{le.classes_[idx]} ({proba[idx]:.1%})"
                    for idx in top3_idx
                ])
                sample["top3_predictions"] = top3_str

                dataset.add_sample(sample)

            except Exception as e:
                print(f"  Error {img_path.name}: {e}")

        print(f"  Cow {cow_id}: loaded")

    print(f"\n{'='*50}")
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Accuracy: {correct}/{total} ({correct/total:.1%})" if total > 0 else "No predictions")
    print(f"{'='*50}")

    # Add dynamic views
    dataset.save()

    # Launch the app
    print(f"\nOpening FiftyOne at http://localhost:5151")
    print("Press Ctrl+C to stop.\n")
    print("Dicas de uso:")
    print("  - Filtre por 'correct == False' para ver erros")
    print("  - Ordene por 'prediction.confidence' para ver predições mais/menos confiantes")
    print("  - Use 'ground_truth.label' para filtrar por vaca")

    session = fo.launch_app(dataset, port=5151)
    session.wait()


if __name__ == "__main__":
    main()
