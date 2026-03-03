import json
import csv
from pathlib import Path
from collections import defaultdict
from core_utils import extract_image_ref, resolve_image_path

def inspect_dataset(input_dir="Key_points", raw_dir="data/raw_images", report_dir="outputs/reports"):
    input_path = Path(input_dir)
    raw_path = Path(raw_dir)
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"Directory {input_dir} not found.")
        return

    json_files = [f for f in input_path.iterdir() if f.is_file()]
    real_images = [f for f in raw_path.iterdir() if f.is_file()] if raw_path.exists() else []
    
    total_files = len(json_files)
    total_images = len(real_images)
    valid_json = 0
    kps_counts = defaultdict(int)
    has_bbox_count = 0
    
    matched_images = 0
    missing_images = []

    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            valid_json += 1
            
            # Match image
            img_ref = extract_image_ref(data)
            real_path, basename = resolve_image_path(img_ref, raw_dir)
            
            if real_path and real_path.exists():
                matched_images += 1
            else:
                missing_images.append({
                    "annotation_file": jf.name,
                    "image_ref": img_ref,
                    "extracted_basename": basename
                })
            
            # Count keypoints and bboxes
            annotations = data.get("annotations", [{}])
            results = data.get("result", [])
            if not results and annotations:
                 results = annotations[0].get("result", [])
            
            has_bbox = False
            for res in results:
                val = res.get("value", {})
                if "rectanglelabels" in val:
                    has_bbox = True
                if "keypointlabels" in val:
                    lbls = val.get("keypointlabels", [])
                    if lbls:
                        kps_counts[lbls[0]] += 1
            
            if has_bbox:
                has_bbox_count += 1
                
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"Error reading {jf}: {e}")

    # Generate CSV of missing images
    csv_path = out_dir / "missing_images.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["annotation_file", "image_ref", "extracted_basename"])
        writer.writeheader()
        writer.writerows(missing_images)

    # Generate Markdown Report
    report_path = out_dir / "dataset_summary.md"
    report = []
    report.append("# Dataset Inspection Report\n")
    report.append(f"- **Total JSON annotations**: {total_files}")
    report.append(f"- **Valid JSONs**: {valid_json}")
    report.append(f"- **Annotations with BBox**: {has_bbox_count}")
    report.append(f"- **Annotations missing BBox**: {valid_json - has_bbox_count}\n")
    
    report.append("## Image Matching\n")
    report.append(f"- **Total real images in `{raw_dir}`**: {total_images}")
    report.append(f"- **Annotations mapped to real images**: {matched_images} ({matched_images/valid_json*100:.1f}%)")
    report.append(f"- **Annotations with missing images**: {len(missing_images)} (Saved to `{csv_path.name}`)\n")
    
    report.append("## Keypoints Distribution\n")
    for kp, count in sorted(kps_counts.items(), key=lambda x: x[1], reverse=True):
        report.append(f"- `{kp}`: {count}")
        
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
        
    print(f"Inspection complete. Repots saved to {out_dir}")

if __name__ == "__main__":
    inspect_dataset()
