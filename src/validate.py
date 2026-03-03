import json
from pathlib import Path

# Mapping requested names to dataset names (based on dataset_summary.md and validate_annotations.py)
KP_MAPPING = {
    "head": "head",
    "neck": "neck",
    "withers": "withers",
    "back": "back",
    "hook": "hook up", # Mapping hook up to hook
    "hip_ridge": "hip", # Mapping hip to hip_ridge
    "tail_head": "tail head",
    "pin": "pin down" # Mapping pin down to pin
}

TARGET_KPS = ["head", "neck", "withers", "back", "hook", "hip_ridge", "tail_head", "pin"]

def validate(input_dir="Key_points", output_report="outputs/reports/validation_report.json"):
    input_path = Path(input_dir)
    output_path = Path(output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    json_files = list(input_path.glob("*"))
    
    report = {
        "total_files": len(json_files),
        "valid_files": 0,
        "files_with_issues": 0,
        "issues": []
    }
    
    for jf in json_files:
        if not jf.is_file():
            continue
            
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            task_id = data.get("id", "Unknown")
            
            annotations = data.get("annotations", [{}])
            results = data.get("result", [])
            if not results and annotations:
                 results = annotations[0].get("result", [])
                 
            has_bbox = False
            found_kps = {}
            
            for res in results:
                val = res.get("value", {})
                if "rectanglelabels" in val:
                    has_bbox = True
                    # Check bbox boundaries
                    x = val.get("x", 0)
                    y = val.get("y", 0)
                    w = val.get("width", 0)
                    h = val.get("height", 0)
                    if x < 0 or y < 0 or x + w > 100 or y + h > 100:
                        report["issues"].append({"file": jf.name, "issue": "BBox out of bounds"})
                        
                if "keypointlabels" in val:
                    lbls = val.get("keypointlabels", [])
                    if lbls:
                        found_kps[lbls[0]] = val
            
            # Count how many target KPs are mapped back to
            mapped_found = 0
            for tkp in TARGET_KPS:
                mapped_name = KP_MAPPING.get(tkp)
                if mapped_name in found_kps:
                    tkp_val = found_kps[mapped_name]
                    tx, ty = tkp_val.get("x", 0), tkp_val.get("y", 0)
                    if tx < 0 or ty < 0 or tx > 100 or ty > 100:
                        report["issues"].append({"file": jf.name, "issue": f"Keypoint {tkp} ({mapped_name}) out of bounds"})
                    mapped_found += 1
            
            file_issues = [i for i in report["issues"] if i["file"] == jf.name]
            
            if not has_bbox:
                report["issues"].append({"file": jf.name, "issue": "Missing BBox"})
                file_issues.append("Missing BBox")
                
            if mapped_found < 3: # Min coherence: at least 3 points
                report["issues"].append({"file": jf.name, "issue": f"Low keypoint count: {mapped_found}/8"})
                file_issues.append("Low KPs")
                
            if file_issues:
                report["files_with_issues"] += 1
            else:
                report["valid_files"] += 1
                
        except json.JSONDecodeError:
            pass
        except Exception as e:
            report["issues"].append({"file": jf.name, "issue": f"Error parsing: {str(e)}"})
            report["files_with_issues"] += 1
            
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
        
    print(f"Validation complete. Valid: {report['valid_files']}, With issues: {report['files_with_issues']}")
    print(f"Report saved to {output_path}")

if __name__ == "__main__":
    validate()
