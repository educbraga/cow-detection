import json
import random
import shutil
from pathlib import Path
import yaml
from core_utils import TARGET_KPS, KP_MAPPING, resolve_image_path, extract_image_ref, parse_annotation_results

def convert_to_yolo_pose(input_dir="Key_points", output_dir="data/processed_yolo_pose", raw_dir="data/raw_images", seed=42):
    random.seed(seed)
    input_path = Path(input_dir)
    out_path = Path(output_dir)
    
    for split in ["train", "val", "test"]:
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    json_files = [f for f in input_path.iterdir() if f.is_file()]
    random.shuffle(json_files)
    
    # We first filter out annotations that don't have matching real images
    valid_samples = []
    
    print("Checking for matching real images...")
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
                    "img_path": real_path,
                    "img_name": real_path.name
                })
        except Exception as e:
            print(f"Error reading {jf.name}: {e}")
            
    n = len(valid_samples)
    print(f"Found {n} valid samples with real images. Splitting...")
    if n == 0:
        print("No valid samples found. Aborting.")
        return
        
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    
    splits = {
        "train": valid_samples[:train_end],
        "val": valid_samples[train_end:val_end],
        "test": valid_samples[val_end:]
    }
    
    Path("outputs/reports").mkdir(exist_ok=True)
    
    for split_name, samples in splits.items():
        with open(f"outputs/reports/split_{split_name}.txt", "w") as f:
            for sample in samples:
                f.write(f"{sample['json_file'].name} -> {sample['img_name']}\n")
                
        for sample in samples:
            try:
                jf = sample["json_file"]
                data = sample["data"]
                img_path = sample["img_path"]
                img_name = sample["img_name"]
                
                results = data.get("result", [])
                annotations = data.get("annotations", [{}])
                if not results and annotations:
                     results = annotations[0].get("result", [])
                
                parsed = parse_annotation_results(results)
                bbox = parsed["bbox"]
                kps_by_name = parsed["keypoints"]
                
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
                    
                line = [0] + bbox # class 0 (cow)
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
                    
                # Copy real image to YOLO dataset
                img_dest = out_path / "images" / split_name / img_name
                if not img_dest.exists():
                    shutil.copy2(img_path, img_dest)
                    
            except Exception as e:
                print(f"Error converting {sample['json_file'].name}: {e}")

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
        
    print(f"Converted dataset to {output_dir}")

if __name__ == "__main__":
    convert_to_yolo_pose()
