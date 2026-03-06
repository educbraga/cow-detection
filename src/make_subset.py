import json
import random
import shutil
from pathlib import Path
import yaml
from core_utils import TARGET_KPS, KP_MAPPING, resolve_image_path, extract_image_ref, parse_annotation_results, validate_annotation, BASE_DIR

def make_subset(input_dir="Key_points", output_dir="data/subset_yolo_pose", raw_dir="data/raw_images", subset_size=150, seed=42):
    random.seed(seed)
    input_path = Path(input_dir)
    out_path = Path(output_dir)
    
    for split in ["train", "val"]:
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    json_files = [f for f in input_path.iterdir() if f.is_file()]
    
    # We first filter out annotations that don't have matching real images
    valid_samples = []
    
    print("Checking for matching real images for subset...")
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
            
    print(f"Dataset loaded: {len(valid_samples)} samples with matching images")
    
    # --- Annotation quality filtering ---
    filtered_samples = []
    rejected_samples = []
    
    for sample in valid_samples:
        data = sample["data"]
        results = data.get("result", [])
        annotations = data.get("annotations", [{}])
        if not results and annotations:
            results = annotations[0].get("result", [])
        
        validation = validate_annotation(results)
        if validation["valid"]:
            filtered_samples.append(sample)
        else:
            sample["issues"] = validation["issues"]
            rejected_samples.append(sample)
    
    n_rejected = len(rejected_samples)
    n_valid = len(filtered_samples)
    print(f"Invalid annotations removed: {n_rejected} (duplicated/missing keypoints)")
    print(f"Final valid samples: {n_valid}")
    
    # Save filtering report
    reports_dir = BASE_DIR / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "filtered_annotations.txt", "w") as f:
        f.write(f"Total with images: {len(valid_samples)}\n")
        f.write(f"Rejected: {n_rejected}\n")
        f.write(f"Valid: {n_valid}\n\n")
        for s in rejected_samples:
            f.write(f"{s['json_file'].name} -> {s['img_name']}: {'; '.join(s['issues'])}\n")
    
    if n_valid == 0:
        print("No valid samples found. Aborting.")
        return
        
    if n_valid < subset_size:
        print(f"Warning: only {n_valid} valid samples, using all instead of {subset_size}.")
        subset_size = n_valid
        
    subset_samples = random.sample(filtered_samples, subset_size)
    
    train_end = int(0.80 * subset_size)
    splits = {
        "train": subset_samples[:train_end],
        "val": subset_samples[train_end:]
    }
    
    with open(reports_dir / "subset_files.txt", "w") as f:
        for sample in subset_samples:
            f.write(f"{sample['json_file'].name} -> {sample['img_name']}\n")
            
    for split_name, samples in splits.items():
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
                    
                line = [0] + bbox # class 0
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
                    
                img_dest = out_path / "images" / split_name / img_name
                if not img_dest.exists():
                    shutil.copy2(img_path, img_dest)
                    
            except Exception as e:
                print(f"Error converting subset {sample['json_file'].name}: {e}")

    yaml_data = {
        "path": str(out_path.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/val",
        "kpt_shape": [8, 3],
        "names": {0: "cow"}
    }
    
    with open(out_path / "data.yaml", "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)
        
    print(f"Created subset of {subset_size} images to {output_dir}")

if __name__ == "__main__":
    from core_utils import BASE_DIR, KEYPOINTS_DIR, RAW_IMAGES_DIR
    make_subset(input_dir=KEYPOINTS_DIR, output_dir=BASE_DIR / "data" / "subset_yolo_pose", raw_dir=RAW_IMAGES_DIR)
