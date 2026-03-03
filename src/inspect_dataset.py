import json
from pathlib import Path
from collections import defaultdict
import urllib.parse
import os

def inspect_dataset(input_dir="Key_points", output_report="outputs/reports/dataset_summary.md"):
    input_path = Path(input_dir)
    output_path = Path(output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"Directory {input_dir} not found.")
        return

    json_files = [f for f in input_path.iterdir() if f.is_file()]
    
    total_files = len(json_files)
    valid_json = 0
    kps_counts = defaultdict(int)
    has_bbox_count = 0
    images_found = []

    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            valid_json += 1
            
            # Extract Image Name
            img_url = data.get("task", {}).get("data", {}).get("img", "")
            if not img_url:
                img_url = data.get("data", {}).get("img", "")
                
            if img_url:
                url_decoded = urllib.parse.unquote(img_url)
                # handle Label Studio local files path like "\Users\Thales Salvador\..."
                img_name = Path(url_decoded.replace('\\', '/')).name
                images_found.append(img_name)
            
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

    # Generate Report
    report = []
    report.append("# Dataset Inspection Report\n")
    report.append(f"- **Total files scanned**: {total_files}")
    report.append(f"- **Valid JSON annotations**: {valid_json}")
    report.append(f"- **Annotations with BBox**: {has_bbox_count}")
    report.append(f"- **Annotations missing BBox**: {valid_json - has_bbox_count}")
    
    report.append("\n## Keypoints Distribution\n")
    for kp, count in sorted(kps_counts.items(), key=lambda x: x[1], reverse=True):
        report.append(f"- `{kp}`: {count}")
        
    report.append(f"\n## Image References\n")
    report.append(f"- Extracted {len(images_found)} image names.")
    if images_found:
        report.append(f"- *Samples*: {', '.join(images_found[:5])}...")

    report_content = "\n".join(report)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print(f"Inspection complete. Report saved to {output_path}")

if __name__ == "__main__":
    inspect_dataset()
