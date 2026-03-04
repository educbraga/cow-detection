import json
import random
import cv2
from pathlib import Path
from core_utils import SKELETON, TARGET_KPS, KP_MAPPING, resolve_image_path, extract_image_ref, parse_annotation_results

def run_sanity_check(input_dir="Key_points", raw_dir="data/raw_images", out_dir="outputs/vis_samples", num_samples=5, seed=42):
    random.seed(seed)
    input_path = Path(input_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    json_files = [f for f in input_path.iterdir() if f.is_file()]
    
    valid_samples = []
    
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            img_ref = extract_image_ref(data)
            real_path, basename = resolve_image_path(img_ref, raw_dir)
            
            if real_path and real_path.exists():
                valid_samples.append({
                    "json_file": jf,
                    "data": data,
                    "img_path": real_path
                })
        except Exception as e:
            pass
            
    if not valid_samples:
        print("No valid samples found for sanity check.")
        return
        
    sample_size = min(num_samples, len(valid_samples))
    samples = random.sample(valid_samples, sample_size)
    
    print(f"Running sanity check on {sample_size} samples...")
    
    for idx, sample in enumerate(samples):
        img_path = sample["img_path"]
        data = sample["data"]
        
        # Parse annotations
        results = data.get("result", [])
        annotations = data.get("annotations", [{}])
        if not results and annotations:
             results = annotations[0].get("result", [])
             
        parsed = parse_annotation_results(results)
        kps_by_name = parsed["keypoints"]
        bbox = parsed["bbox"] # [cx, cy, w, h] normalized
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h_img, w_img = img.shape[:2]
        
        # Draw BBox
        if bbox:
            cx, cy, bw, bh = bbox
            # Denormalize
            cx_px, cy_px = int(cx * w_img), int(cy * h_img)
            bw_px, bh_px = int(bw * w_img), int(bh * h_img)
            
            x1 = max(0, cx_px - bw_px // 2)
            y1 = max(0, cy_px - bh_px // 2)
            x2 = min(w_img - 1, cx_px + bw_px // 2)
            y2 = min(h_img - 1, cy_px + bh_px // 2)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "GT BBox", (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # Draw Keypoints
        pts = []
        for i, tkp in enumerate(TARGET_KPS):
            mapped_name = KP_MAPPING.get(tkp)
            if mapped_name in kps_by_name:
                kx, ky, v = kps_by_name[mapped_name]
                # Denormalize
                x_px = int(kx * w_img)
                y_px = int(ky * h_img)
                pts.append((x_px, y_px))
                cv2.circle(img, (x_px, y_px), 8, (0, 0, 255), -1)
                cv2.putText(img, tkp, (x_px+10, y_px-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                pts.append(None)
                
        # Draw Skeleton
        for sk in SKELETON:
            pt1 = pts[sk[0]]
            pt2 = pts[sk[1]]
            if pt1 is not None and pt2 is not None:
                cv2.line(img, pt1, pt2, (255, 0, 0), 3)
                
        out_file = out_path / f"sanity_gt_{img_path.name}"
        cv2.imwrite(str(out_file), img)
        print(f"Saved sanity sample: {out_file}")

if __name__ == "__main__":
    from core_utils import KEYPOINTS_DIR, RAW_IMAGES_DIR, OUTPUTS_DIR
    run_sanity_check(input_dir=KEYPOINTS_DIR, raw_dir=RAW_IMAGES_DIR, out_dir=OUTPUTS_DIR / "vis_samples")
