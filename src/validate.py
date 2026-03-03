import json
from pathlib import Path
from core_utils import TARGET_KPS, KP_MAPPING, resolve_image_path, extract_image_ref

def validate(input_dir="Key_points", raw_dir="data/raw_images", output_report="outputs/reports/validation_report.json"):
    input_path = Path(input_dir)
    output_path = Path(output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    json_files = [f for f in input_path.iterdir() if f.is_file()]
    
    report = {
        "total_files": len(json_files),
        "valid_files": 0,
        "files_with_issues": 0,
        "issues": []
    }
    
    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            img_ref = extract_image_ref(data)
            real_path, basename = resolve_image_path(img_ref, raw_dir)
            
            results = data.get("result", [])
            annotations = data.get("annotations", [{}])
            if not results and annotations:
                 results = annotations[0].get("result", [])
                 
            has_bbox = False
            found_kps = {}
            unknown_labels = set()
            
            for res in results:
                val = res.get("value", {})
                if "rectanglelabels" in val:
                    has_bbox = True
                    x = val.get("x", 0)
                    y = val.get("y", 0)
                    w = val.get("width", 0)
                    h = val.get("height", 0)
                    # Use -0.01 and 100.01 to allow small floating point inaccuracies like -1.16e-15
                    if x < -0.01 or y < -0.01 or x + w > 100.01 or y + h > 100.01:
                        report["issues"].append({"file": jf.name, "issue": "BBox out of bounds (0-100%)"})
                        
                if "keypointlabels" in val:
                    lbls = val.get("keypointlabels", [])
                    if lbls:
                        kp_name = lbls[0]
                        found_kps[kp_name] = val
                        if kp_name not in KP_MAPPING.values():
                            unknown_labels.add(kp_name)
            
            file_issues = [i for i in report["issues"] if i["file"] == jf.name]
            
            if unknown_labels:
                issue_msg = f"Unknown labels found: {', '.join(unknown_labels)}"
                report["issues"].append({"file": jf.name, "issue": issue_msg})
                file_issues.append(issue_msg)
                
            if not real_path:
                issue_msg = f"Image not found in raw_images: {basename}"
                report["issues"].append({"file": jf.name, "issue": issue_msg})
                file_issues.append(issue_msg)
            
            mapped_found = []
            for tkp in TARGET_KPS:
                mapped_name = KP_MAPPING.get(tkp)
                if mapped_name in found_kps:
                    tkp_val = found_kps[mapped_name]
                    tx, ty = tkp_val.get("x", 0), tkp_val.get("y", 0)
                    if tx < -0.01 or ty < -0.01 or tx > 100.01 or ty > 100.01:
                        report["issues"].append({"file": jf.name, "issue": f"Keypoint {tkp} ({mapped_name}) out of bounds"})
                    mapped_found.append(tkp)
            
            missing_kps = [k for k in TARGET_KPS if k not in mapped_found]
            
            if not has_bbox:
                report["issues"].append({"file": jf.name, "issue": "Missing BBox"})
                file_issues.append("Missing BBox")
                
            if missing_kps:
                report["issues"].append({"file": jf.name, "issue": f"Missing keypoints ({len(missing_kps)}): {', '.join(missing_kps)}"})
                file_issues.append("Missing KPs")
                
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
