"""
Debug Visualization Script for YOLO Pose Dataset
Draws bounding boxes AND keypoints on images from the YOLO dataset.
Also compares with Label Studio ground truth when available.
"""
import json
import random
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from core_utils import TARGET_KPS, SKELETON, BASE_DIR, extract_image_ref, resolve_image_path, parse_annotation_results

COLORS = [
    (0, 0, 255),    # withers - red
    (0, 128, 255),  # back - orange
    (0, 255, 255),  # hook up - yellow
    (0, 255, 0),    # hook down - green  
    (255, 255, 0),  # hip - cyan
    (255, 0, 0),    # tail head - blue
    (255, 0, 128),  # pin up - purple
    (128, 0, 255),  # pin down - magenta
]

def draw_yolo_label(img, label_path, title="YOLO Label"):
    """Draw bounding box and keypoints from a YOLO label file."""
    h, w = img.shape[:2]
    result = img.copy()
    
    with open(label_path) as f:
        line = f.readline().strip().split()
    
    # Parse bbox
    cls = int(line[0])
    xc, yc, bw, bh = [float(v) for v in line[1:5]]
    
    x1 = int((xc - bw/2) * w)
    y1 = int((yc - bh/2) * h)
    x2 = int((xc + bw/2) * w)
    y2 = int((yc + bh/2) * h)
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Parse keypoints
    kp_tokens = line[5:]
    kps = []
    for i in range(0, len(kp_tokens), 3):
        kx = float(kp_tokens[i]) * w
        ky = float(kp_tokens[i+1]) * h
        v = int(kp_tokens[i+2])
        kps.append((int(kx), int(ky), v))
    
    # Draw keypoints
    for i, (kx, ky, v) in enumerate(kps):
        if v > 0 and kx > 0 and ky > 0:
            color = COLORS[i % len(COLORS)]
            cv2.circle(result, (kx, ky), 6, color, -1)
            cv2.circle(result, (kx, ky), 8, (255, 255, 255), 1)
            name = TARGET_KPS[i] if i < len(TARGET_KPS) else str(i)
            cv2.putText(result, f"{i}:{name}", (kx + 8, ky - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    cv2.putText(result, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return result, kps


def draw_label_studio_gt(img, json_path, title="Label Studio GT"):
    """Draw keypoints from Label Studio JSON annotation."""
    h, w = img.shape[:2]
    result = img.copy()
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = data.get("result", [])
    if not results:
        annotations = data.get("annotations", [{}])
        if annotations:
            results = annotations[0].get("result", [])
    
    parsed = parse_annotation_results(results)
    kps_by_name = parsed["keypoints"]
    bbox = parsed["bbox"]
    
    # Draw bbox
    if bbox:
        xc, yc, bw, bh = bbox
        x1 = int((xc - bw/2) * w)
        y1 = int((yc - bh/2) * h)
        x2 = int((xc + bw/2) * w)
        y2 = int((yc + bh/2) * h)
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw keypoints in TARGET_KPS order
    kps = []
    for i, name in enumerate(TARGET_KPS):
        if name in kps_by_name:
            kx, ky, v = kps_by_name[name]
            px, py = int(kx * w), int(ky * h)
            kps.append((px, py, v))
            if v > 0:
                color = COLORS[i % len(COLORS)]
                cv2.circle(result, (px, py), 6, color, -1)
                cv2.circle(result, (px, py), 8, (255, 255, 255), 1)
                cv2.putText(result, f"{i}:{name}", (px + 8, py - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        else:
            kps.append((0, 0, 0))
    
    cv2.putText(result, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return result, kps


def find_matching_json(img_name):
    """Find the Label Studio JSON for a given image name."""
    kp_dir = BASE_DIR / "Key_points"
    for jf in kp_dir.iterdir():
        if not jf.is_file():
            continue
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            img_ref = extract_image_ref(data)
            _, basename = resolve_image_path(img_ref)
            if basename and basename.lower() == img_name.lower():
                return jf
        except:
            continue
    return None


def main():
    out_dir = BASE_DIR / "outputs" / "vis_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Pick 3 random samples from train
    img_dir = BASE_DIR / "data" / "subset_yolo_pose" / "images" / "train"
    label_dir = BASE_DIR / "data" / "subset_yolo_pose" / "labels" / "train"
    
    images = list(img_dir.glob("*.jpg"))
    if not images:
        print("No images found!")
        return
    
    samples = random.sample(images, min(3, len(images)))
    
    for img_path in samples:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read {img_path.name}")
            continue
        
        label_path = label_dir / img_path.with_suffix(".txt").name
        if not label_path.exists():
            print(f"No label for {img_path.name}")
            continue
        
        # Draw YOLO label
        yolo_img, yolo_kps = draw_yolo_label(img, label_path, "YOLO Label (bbox + kps only)")
        
        # Try to find matching Label Studio JSON
        json_path = find_matching_json(img_path.name)
        
        if json_path:
            ls_img, ls_kps = draw_label_studio_gt(img, json_path, "Label Studio GT")
            
            # Side by side comparison
            combined = np.hstack([ls_img, yolo_img])
            
            # Print coordinate comparison
            print(f"\n=== {img_path.name} ===")
            print(f"{'KP Name':<12} {'LS (px)':<20} {'YOLO (px)':<20} {'Match?'}")
            for i, name in enumerate(TARGET_KPS):
                if i < len(ls_kps) and i < len(yolo_kps):
                    ls_pt = ls_kps[i]
                    yolo_pt = yolo_kps[i]
                    match = abs(ls_pt[0]-yolo_pt[0]) <= 2 and abs(ls_pt[1]-yolo_pt[1]) <= 2
                    print(f"{name:<12} ({ls_pt[0]:>4},{ls_pt[1]:>4}) v={ls_pt[2]}   ({yolo_pt[0]:>4},{yolo_pt[1]:>4}) v={yolo_pt[2]}   {'✓' if match else '✗ MISMATCH'}")
        else:
            combined = yolo_img
            print(f"\n=== {img_path.name} === (no matching Label Studio JSON found)")
        
        out_file = out_dir / f"debug_{img_path.name}"
        cv2.imwrite(str(out_file), combined)
        print(f"  Saved: {out_file}")
    
    print(f"\nDone! Check {out_dir} for debug images.")


if __name__ == "__main__":
    main()
