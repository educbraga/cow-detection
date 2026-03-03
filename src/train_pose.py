import argparse
import os
import shutil
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLO Pose model for Cow Detection")
    parser.add_argument("--data", type=str, default="data/subset_yolo_pose/data.yaml", 
                        help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Number of epochs to train (default 10 for subset, use 100 for full)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", type=str, default="yolo11n-pose.pt", 
                        help="Model weights. Will try yolo26-pose.pt if available.")
    
    args = parser.parse_args()
    
    # Check for yolo26 pose fallback
    model_name = args.model
    if not Path(model_name).exists():
        if "yolo26" in model_name:
            print(f"Warning: {model_name} not found. Falling back to yolo8n-pose.pt (since 11n-pose.pt might not exist depending on Ultralytics version)")
            model_name = "yolov8n-pose.pt" # Safe fallback for widely supported versions
            
    print(f"Initializing training with model: {model_name}")
    print(f"Data config: {args.data}")
    print(f"Epochs: {args.epochs}")
    
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Failed to load {model_name}. Attempting yolov8n-pose.pt as safe fallback.")
        model = YOLO("yolov8n-pose.pt")
        
    output_project = "outputs/runs/train"
    run_name = "cow_pose_run"
    
    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        seed=args.seed,
        project=output_project,
        name=run_name,
        exist_ok=True, # Overwrite run if rerun
        device="cpu" # To ensure it runs everywhere, or omit for auto
    )
    
    # Save exported best model to outputs/models
    try:
        best_model_path = Path(output_project) / run_name / "weights" / "best.pt"
        if best_model_path.exists():
            dest = Path("outputs/models/best_pose.pt")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_model_path, dest)
            print(f"Best model saved to {dest}")
    except Exception as e:
        print(f"Failed to copy best model to outputs/models: {e}")

if __name__ == "__main__":
    main()
