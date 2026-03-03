import json
import random
from pathlib import Path
import urllib.parse
import yaml

TARGET_KPS = ["head", "neck", "withers", "back", "hook", "hip_ridge", "tail_head", "pin"]
KP_MAPPING = {
    "head": "head",
    "neck": "neck",
    "withers": "withers",
    "back": "back",
    "hook": "hook up",
    "hip_ridge": "hip",
    "tail_head": "tail head",
    "pin": "pin down"
}

def convert_to_yolo_pose(input_dir="Key_points", output_dir="data/processed_yolo_pose", seed=42):
    random.seed(seed)
    input_path = Path(input_dir)
    out_path = Path(output_dir)
    
    for split in ["train", "val", "test"]:
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    json_files = [f for f in input_path.iterdir() if f.is_file()]
    random.shuffle(json_files)
    n = len(json_files)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    
    splits = {
        "train": json_files[:train_end],
        "val": json_files[train_end:val_end],
        "test": json_files[val_end:]
    }
    
    Path("outputs/reports").mkdir(exist_ok=True)
    
    for split_name, files in splits.items():
        with open(f"outputs/reports/split_{split_name}.txt", "w") as f:
            for jf in files:
                f.write(jf.name + "\n")
                
        for jf in files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                img_url = data.get("task", {}).get("data", {}).get("img", "")
                if not img_url:
                    img_url = data.get("data", {}).get("img", "")
                    
                img_name = Path(urllib.parse.unquote(img_url).replace('\\', '/')).name if img_url else f"{jf.name}.jpg"
                
                annotations = data.get("annotations", [{}])
                results = data.get("result", [])
                if not results and annotations:
                     results = annotations[0].get("result", [])
                
                found_kps = {}
                bbox = None
                
                # First pass: find keypoints and bboxes
                for res in results:
                    val = res.get("value", {})
                    if res.get("type") == "rectanglelabels":
                        x = val.get("x", 0) / 100.0
                        y = val.get("y", 0) / 100.0
                        w = val.get("width", 0) / 100.0
                        h = val.get("height", 0) / 100.0
                        bbox = [x + w/2, y + h/2, w, h] 
                        
                    elif res.get("type") == "keypointlabels":
                        lbls = val.get("keypointlabels", [])
                        if lbls:
                            kp_name = lbls[0]
                            kx = val.get("x", 0) / 100.0
                            ky = val.get("y", 0) / 100.0
                            found_kps[res.get("id")] = {"name": kp_name, "x": kx, "y": ky, "v": 2} # default visible
                            
                # Second pass: find visibility choices linked by id
                for res in results:
                    if res.get("type") == "choices" and res.get("from_name") == "visibility":
                        val = res.get("value", {})
                        choices = val.get("choices", [])
                        if choices and res.get("id") in found_kps:
                            # 2 = visible, 1 = occluded
                            visibility_val = 1 if choices[0].lower() == "oculto" else 2
                            found_kps[res.get("id")]["v"] = visibility_val
                            
                # Reorganize by name
                kps_by_name = {kp["name"]: (kp["x"], kp["y"], kp["v"]) for kp in found_kps.values()}
                            
                # Infer bbox if not present
                if not bbox and kps_by_name:
                    xs = [p[0] for p in kps_by_name.values()]
                    ys = [p[1] for p in kps_by_name.values()]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    
                    w = (max_x - min_x) * 1.15
                    h = (max_y - min_y) * 1.15
                    cx = (min_x + max_x) / 2
                    cy = (min_y + max_y) / 2
                    
                    # Clamp bounding box coords to [0, 1]
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    w = min(1.0, w)
                    h = min(1.0, h)
                    
                    bbox = [cx, cy, w, h]
                    
                if not bbox:
                    continue
                    
                line = [0] + bbox
                for tkp in TARGET_KPS:
                    mapped_name = KP_MAPPING.get(tkp)
                    if mapped_name in kps_by_name:
                        kx, ky, v = kps_by_name[mapped_name]
                        line.extend([kx, ky, v])
                    else:
                        line.extend([0.0, 0.0, 0])
                        
                label_str = " ".join([f"{v:.6f}" if isinstance(v, float) else str(v) for v in line])
                label_filename = Path(img_name).with_suffix(".txt").name
                
                with open(out_path / "labels" / split_name / label_filename, "w") as lf:
                    lf.write(label_str + "\n")
                    
                # Ensure an image exists to satisfy YOLO
                img_dest = out_path / "images" / split_name / img_name
                import shutil
                if not img_dest.exists():
                    img_src = input_path / img_name
                    if img_src.exists():
                        shutil.copy2(img_src, img_dest)
                    else:
                        img_src_alt = input_path.parent / "images" / img_name
                        if img_src_alt.exists():
                            shutil.copy2(img_src_alt, img_dest)
                        else:
                            # Create dummy image
                            import numpy as np
                            import cv2
                            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                            cv2.imwrite(str(img_dest), dummy_img)
                    
            except Exception as e:
                print(f"Error converting {jf.name}: {e}")

    yaml_data = {
        "path": str(out_path.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "kpt_shape": [8, 3],
        "names": {0: "cow"}
    }
    
    with open(out_path / "data.yaml", "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)
        
    print(f"Converted full dataset to {output_dir}")

if __name__ == "__main__":
    convert_to_yolo_pose()
