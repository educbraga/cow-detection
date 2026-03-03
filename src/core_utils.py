import re
import urllib.parse
from pathlib import Path

TARGET_KPS = ["withers", "back", "hook up", "hook down", "hip", "tail head", "pin up", "pin down"]

KP_MAPPING = {k: k for k in TARGET_KPS}

SKELETON = [
    [0, 1], # withers-back
    [1, 2], # back-hook up
    [2, 3], # hook up-hook down
    [2, 4], # hook up-hip
    [4, 5], # hip-tail head
    [5, 6], # tail head-pin up
    [6, 7]  # pin up-pin down
]

def parse_filename_metadata(filename):
    """
    Parses cow detection filenames.
    """
    name = filename
    if name.startswith("RLC4_00_"):
        name = name[8:]
        
    pattern = r"^(?P<date>\d{8})_(?P<time>\d{6})_(?P<station>baia\d+)_(?P<cam>[A-Za-z0-9]+)\.jpe?g$"
    match = re.match(pattern, name, re.IGNORECASE)
    
    if match:
        data = match.groupdict()
        data["cow_id"] = None
        return data
    
    return None

def extract_image_ref(data):
    """
    Extracts the image URL from a Label Studio JSON annotation.
    """
    img_url = data.get("task", {}).get("data", {}).get("img", "")
    if not img_url:
        img_url = data.get("data", {}).get("img", "")
    return img_url

def resolve_image_path(image_ref, raw_dir="data/raw_images"):
    """
    Takes an image reference from Label Studio and attempts to find it in the raw_dir.
    """
    if not image_ref:
        return None, None
        
    url_decoded = urllib.parse.unquote(image_ref).replace('\\', '/')
    basename = Path(url_decoded).name
    
    # Strip the UUID if present (format: 8 chars + dash)
    clean_name = basename
    uuid_pattern = r"^[a-fA-F0-9]{8}-(.+\.jpe?g)$"
    match = re.match(uuid_pattern, basename, re.IGNORECASE)
    if match:
        clean_name = match.group(1)
        
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return None, clean_name
        
    # Standard lookup
    target_path = raw_path / clean_name
    if target_path.exists():
        return target_path, clean_name
        
    target_path_with_uuid = raw_path / basename
    if target_path_with_uuid.exists():
        return target_path_with_uuid, basename
        
    # Recursive lookup
    for p in raw_path.rglob("*"):
        if p.is_file():
            if p.name.lower() == clean_name.lower():
                return p, p.name
                
    return None, clean_name

def parse_annotation_results(results):
    """
    Parses 'result' list from Label Studio JSON into a dict with bbox and keypoints.
    """
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
    return {"bbox": bbox, "keypoints": kps_by_name}
