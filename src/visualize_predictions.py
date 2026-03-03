import argparse
from pathlib import Path
import random
import cv2
from ultralytics import YOLO
import sys

# Local imports for shared constants
from core_utils import TARGET_KPS, SKELETON

def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO Pose predictions")
    parser.add_argument("--model", type=str, default="outputs/models/best_pose.pt", help="Path to best.pt")
    parser.add_argument("--data-dir", type=str, default="data/subset_yolo_pose/images/val", help="Test images dir")
    parser.add_argument("--num-samples", type=int, default=30, help="Number of images to visualize")
    args = parser.parse_args()
    
    out_dir = Path("outputs/vis_samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    img_dir = Path(args.data_dir)
    if not img_dir.exists():
        print(f"Data dir {img_dir} not found. Ensure dataset is converted.")
        return
        
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not images:
        print(f"No images in {img_dir}.")
        return
        
    if len(images) > args.num_samples:
        images = random.sample(images, args.num_samples)
        
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Could not load model {args.model}: {e}")
        return
        
    print(f"Generating visualizations for {len(images)} images in {out_dir}")
    
    for img_path in images:
        try:
            results = model.predict(source=str(img_path), save=False, verbose=False)
            
            img = cv2.imread(str(img_path))
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                keypoints = result.keypoints
                if keypoints is not None and keypoints.xy is not None and len(keypoints.xy) > 0:
                    for kp_set in keypoints.xy:
                        pts = []
                        for i, pt in enumerate(kp_set):
                            x, y = map(int, pt)
                            if x > 0 and y > 0:
                                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                                kp_name = TARGET_KPS[i] if i < len(TARGET_KPS) else str(i)
                                cv2.putText(img, kp_name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                            pts.append((x, y))
                            
                        for sk in SKELETON:
                            if sk[0] < len(pts) and sk[1] < len(pts):
                                pt1 = pts[sk[0]]
                                pt2 = pts[sk[1]]
                                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                                    cv2.line(img, pt1, pt2, (255, 0, 0), 2)
                                
            out_file = out_dir / f"pred_{img_path.name}"
            cv2.imwrite(str(out_file), img)
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            
    print("Visualizations complete.")

if __name__ == "__main__":
    main()
