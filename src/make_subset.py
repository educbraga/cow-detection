import json
import random
from pathlib import Path
import urllib.parse
import yaml
import shutil

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

def make_subset(input_dir="Key_points", output_dir="data/subset_yolo_pose", subset_size=150, seed=42):
    random.seed(seed)
    input_path = Path(input_dir)
    out_path = Path(output_dir)
    
    for split in ["train", "val"]:
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    json_files = [f for f in input_path.iterdir() if f.is_file()]
    
    if len(json_files) < subset_size:
        print(f"Warning: only {len(json_files)} found, using all of them.")
        subset_size = len(json_files)
        
    subset_files = random.sample(json_files, subset_size)
    
    train_end = int(0.80 * subset_size)
    splits = {
        "train": subset_files[:train_end],
        "val": subset_files[train_end:]
    }
    
    Path("outputs/reports").mkdir(exist_ok=True)
    with open("outputs/reports/subset_files.txt", "w") as f:
        for jf in subset_files:
            f.write(jf.name + "\n")
            
    for split_name, files in splits.items():
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
                            found_kps[res.get("id")] = {"name": kp_name, "x": kx, "y": ky, "v": 2}
                            
                for res in results:
                    if res.get("type") == "choices" and res.get("from_name") == "visibility":
                        val = res.get("value", {})
                        choices = val.get("choices", [])
                        if choices and res.get("id") in found_kps:
                            visibility_val = 1 if choices[0].lower() == "oculto" else 2
                            found_kps[res.get("id")]["v"] = visibility_val
                            
                kps_by_name = {kp["name"]: (kp["x"], kp["y"], kp["v"]) for kp in found_kps.values()}
                            
                if not bbox and kps_by_name:
                    xs = [p[0] for p in kps_by_name.values()]
                    ys = [p[1] for p in kps_by_name.values()]
                    bbox = [
                        max(0.0, min(1.0, (min(xs) + max(xs)) / 2)),
                        max(0.0, min(1.0, (min(ys) + max(ys)) / 2)),
                        min(1.0, (max(xs) - min(xs)) * 1.15),
                        min(1.0, (max(ys) - min(ys)) * 1.15)
                    ]
                    
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
                if not img_dest.exists():
                    img_src = input_path / img_name
                    if img_src.exists():
                        shutil.copy2(img_src, img_dest)
                    else:
                        img_src_alt = input_path.parent / "images" / img_name
                        if img_src_alt.exists():
                            shutil.copy2(img_src_alt, img_dest)
                        else:
                            # Create dummy 640x640 black image so YOLO doesn't crash on "No images found"
                            import numpy as np
                            import cv2
                            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                            cv2.imwrite(str(img_dest), dummy_img)
                    
            except Exception as e:
                print(f"Error converting subset {jf.name}: {e}")

    yaml_data = {
        "path": str(out_path.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/val", # No test split in subset, validate on val
        "kpt_shape": [8, 3],
        "names": {0: "cow"}
    }
    
    with open(out_path / "data.yaml", "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)
        
    print(f"Created subset of {subset_size} images to {output_dir}")

if __name__ == "__main__":
    make_subset()
