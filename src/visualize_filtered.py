"""
Visualizes rejected annotations to confirm the filtering logic is correct.
Shows duplicate keypoints highlighted on the image.
"""
import json
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from core_utils import (
    TARGET_KPS, KEYPOINTS_DIR, RAW_IMAGES_DIR, BASE_DIR,
    extract_image_ref, resolve_image_path, validate_annotation
)

COLORS = [
    (0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0),
    (255, 255, 0), (255, 0, 0), (255, 0, 128), (128, 0, 255),
]


def draw_all_keypoints(img, results, issues_text):
    """Draw ALL keypoint instances (including duplicates) on the image."""
    h, w = img.shape[:2]
    result = img.copy()

    kp_instances = {}
    for res in results:
        if res.get("type") == "keypointlabels":
            val = res.get("value", {})
            lbls = val.get("keypointlabels", [])
            if lbls:
                name = lbls[0]
                kx = val.get("x", 0) / 100.0
                ky = val.get("y", 0) / 100.0
                if name not in kp_instances:
                    kp_instances[name] = []
                kp_instances[name].append((int(kx * w), int(ky * h)))

    for name, points in kp_instances.items():
        idx = TARGET_KPS.index(name) if name in TARGET_KPS else 0
        color = COLORS[idx % len(COLORS)]
        is_dup = len(points) > 1

        for i, (px, py) in enumerate(points):
            if is_dup:
                # Draw duplicates with a thick magenta ring
                cv2.circle(result, (px, py), 12, (255, 0, 255), 3)
            cv2.circle(result, (px, py), 6, color, -1)
            label = f"{name}" if not is_dup else f"{name} #{i+1}"
            cv2.putText(result, label, (px + 10, py - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Issues banner
    cv2.putText(result, "REJECTED - " + issues_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return result


def main():
    out_dir = BASE_DIR / "outputs" / "vis_filtered"
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    max_vis = 10

    for jf in sorted(KEYPOINTS_DIR.iterdir()):
        if not jf.is_file() or count >= max_vis:
            break
        try:
            data = json.load(open(jf, "r", encoding="utf-8"))
            results = data.get("result", [])
            if not results:
                annotations = data.get("annotations", [{}])
                if annotations:
                    results = annotations[0].get("result", [])

            validation = validate_annotation(results)
            if validation["valid"]:
                continue

            img_ref = extract_image_ref(data)
            img_path, img_name = resolve_image_path(img_ref)
            if not img_path or not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            issues_text = "; ".join(validation["issues"])
            vis = draw_all_keypoints(img, results, issues_text)

            out_file = out_dir / f"filtered_{jf.name}_{img_name}"
            cv2.imwrite(str(out_file), vis)
            print(f"Saved: {out_file.name} | {issues_text}")
            count += 1

        except Exception as e:
            pass

    print(f"\nDone! Saved {count} filtered visualizations to {out_dir}")


if __name__ == "__main__":
    main()
