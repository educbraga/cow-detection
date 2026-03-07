"""
Extract geometric features from YOLO Pose predictions.
Runs inference on all images, computes angles/distances/proportions,
and saves a features table (CSV).
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core_utils import TARGET_KPS, parse_filename_metadata, BASE_DIR

# Indices for each keypoint in the prediction array (same order as TARGET_KPS)
KP_NAMES = TARGET_KPS  # ["withers", "back", "hook up", "hook down", "hip", "tail head", "pin up", "pin down"]

def angle_between(a, b, c):
    """Angle at point b formed by segments ba and bc (in degrees)."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))

def distance(a, b):
    """Euclidean distance between two points."""
    return np.linalg.norm(np.array(a) - np.array(b))

def extract_features_from_keypoints(kps_dict):
    """
    Given a dict of {kp_name: (x, y)}, compute geometric features.
    Returns a dict of feature_name: value.
    """
    features = {}
    
    # Raw keypoint coordinates
    for name in KP_NAMES:
        if name in kps_dict:
            features[f"kp_{name.replace(' ', '_')}_x"] = kps_dict[name][0]
            features[f"kp_{name.replace(' ', '_')}_y"] = kps_dict[name][1]
        else:
            features[f"kp_{name.replace(' ', '_')}_x"] = np.nan
            features[f"kp_{name.replace(' ', '_')}_y"] = np.nan
    
    # Angles (inspired by the paper's neck/withers/back/hip angles)
    # Angle at "back" formed by withers-back-hip
    if all(k in kps_dict for k in ["withers", "back", "hip"]):
        features["angle_withers_back_hip"] = angle_between(
            kps_dict["withers"], kps_dict["back"], kps_dict["hip"]
        )
    else:
        features["angle_withers_back_hip"] = np.nan
    
    # Angle at "hip" formed by back-hip-tail_head
    if all(k in kps_dict for k in ["back", "hip", "tail head"]):
        features["angle_back_hip_tail"] = angle_between(
            kps_dict["back"], kps_dict["hip"], kps_dict["tail head"]
        )
    else:
        features["angle_back_hip_tail"] = np.nan
    
    # Angle at "tail head" formed by hip-tail_head-pin_up (or pin_down)
    if all(k in kps_dict for k in ["hip", "tail head", "pin up"]):
        features["angle_hip_tail_pin"] = angle_between(
            kps_dict["hip"], kps_dict["tail head"], kps_dict["pin up"]
        )
    else:
        features["angle_hip_tail_pin"] = np.nan
    
    # Hook angle: hook_up - hip - hook_down
    if all(k in kps_dict for k in ["hook up", "hip", "hook down"]):
        features["angle_hook_up_hip_hook_down"] = angle_between(
            kps_dict["hook up"], kps_dict["hip"], kps_dict["hook down"]
        )
    else:
        features["angle_hook_up_hip_hook_down"] = np.nan
    
    # Pin angle: pin_up - tail_head - pin_down
    if all(k in kps_dict for k in ["pin up", "tail head", "pin down"]):
        features["angle_pin_up_tail_pin_down"] = angle_between(
            kps_dict["pin up"], kps_dict["tail head"], kps_dict["pin down"]
        )
    else:
        features["angle_pin_up_tail_pin_down"] = np.nan
    
    # Distances
    dist_pairs = [
        ("withers", "back"),
        ("back", "hip"),
        ("hip", "tail head"),
        ("tail head", "pin up"),
        ("tail head", "pin down"),
        ("hook up", "hook down"),
        ("pin up", "pin down"),
        ("withers", "tail head"),  # total spine length
        ("withers", "hip"),       # front spine
    ]
    
    for a, b in dist_pairs:
        fname = f"dist_{a.replace(' ', '_')}_{b.replace(' ', '_')}"
        if a in kps_dict and b in kps_dict:
            features[fname] = distance(kps_dict[a], kps_dict[b])
        else:
            features[fname] = np.nan
    
    # Proportions (normalized by total spine length: withers → tail head)
    spine_len = features.get("dist_withers_tail_head", np.nan)
    if not np.isnan(spine_len) and spine_len > 1e-6:
        for key in list(features.keys()):
            if key.startswith("dist_") and key != "dist_withers_tail_head":
                features[f"ratio_{key[5:]}"] = features[key] / spine_len
    
    # Width ratios
    hook_width = features.get("dist_hook_up_hook_down", np.nan)
    pin_width = features.get("dist_pin_up_pin_down", np.nan)
    if not np.isnan(hook_width) and not np.isnan(pin_width) and pin_width > 1e-6:
        features["ratio_hook_to_pin_width"] = hook_width / pin_width
    else:
        features["ratio_hook_to_pin_width"] = np.nan
    
    return features


def main():
    parser = argparse.ArgumentParser(description="Extract features from YOLO Pose predictions")
    parser.add_argument("--model", type=str, default="outputs/models/best_pose.pt", help="Path to trained pose model")
    parser.add_argument("--images", type=str, default="data/raw_images", help="Directory of images to run inference on")
    parser.add_argument("--output", type=str, default="data/processed/features.csv", help="Output CSV path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run train_pose.py first.")
        sys.exit(1)
    
    model = YOLO(str(model_path))
    images_dir = Path(args.images)
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")))
    
    print(f"Running inference on {len(image_files)} images...")
    
    rows = []
    for img_path in image_files:
        try:
            results = model(str(img_path), conf=args.conf, verbose=False)
            
            if not results or len(results) == 0:
                continue
            
            result = results[0]
            if result.keypoints is None or len(result.keypoints) == 0:
                continue
            
            # Take the detection with highest confidence
            if result.boxes is not None and len(result.boxes) > 0:
                best_idx = result.boxes.conf.argmax().item()
            else:
                best_idx = 0
            
            kps_data = result.keypoints.data[best_idx].cpu().numpy()  # shape: (8, 3) -> x, y, conf
            
            # Build kps_dict with only high-confidence keypoints
            kps_dict = {}
            for i, name in enumerate(KP_NAMES):
                if i < len(kps_data):
                    x, y, conf = kps_data[i]
                    if conf > 0.3:  # Minimum keypoint confidence
                        kps_dict[name] = (float(x), float(y))
            
            if len(kps_dict) < 3:  # Need at least 3 keypoints
                continue
            
            # Extract features
            features = extract_features_from_keypoints(kps_dict)
            
            # Parse filename metadata
            meta = parse_filename_metadata(img_path.name)
            features["image"] = img_path.name
            features["cow_id"] = meta.get("cow_id") if meta else None
            features["station"] = meta.get("station") if meta else None
            features["cam"] = meta.get("cam") if meta else None
            features["date"] = meta.get("date") if meta else None
            features["time"] = meta.get("time") if meta else None
            features["n_keypoints_detected"] = len(kps_dict)
            
            rows.append(features)
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
    
    if not rows:
        print("No features extracted. Check model and images.")
        sys.exit(1)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns: metadata first, then features
    meta_cols = ["image", "cow_id", "station", "cam", "date", "time", "n_keypoints_detected"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + feature_cols]
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nFeatures extracted: {len(df)} images")
    print(f"Columns: {len(df.columns)}")
    print(f"Saved to: {output_path}")
    print(f"\nFeature summary:")
    print(df.describe().to_string())


if __name__ == "__main__":
    main()
