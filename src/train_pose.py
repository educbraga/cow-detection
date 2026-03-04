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
    
    # --- Exportar melhor modelo para outputs/models/best_pose.pt ---
    save_dir = Path(str(getattr(results, 'save_dir', Path(output_project) / run_name)))
    print(f"\nRun save_dir: {save_dir}")

    if not save_dir.exists():
        print(f"ERRO: Diretório do run não encontrado: {save_dir}")
        print("O treino pode ter falhado. Verifique os logs acima.")
        return

    weights_dir = save_dir / "weights"
    weight_src = None

    # 1) best.pt
    if (weights_dir / "best.pt").exists():
        weight_src = weights_dir / "best.pt"
    # 2) last.pt
    elif (weights_dir / "last.pt").exists():
        weight_src = weights_dir / "last.pt"
    # 3) fallback: glob
    elif weights_dir.exists():
        best_glob = sorted(weights_dir.glob("best*.pt"))
        if best_glob:
            weight_src = best_glob[0]
        else:
            all_pts = sorted(weights_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if all_pts:
                weight_src = all_pts[0]

    if weight_src is None:
        print(f"ERRO: Nenhum peso (.pt) encontrado em {weights_dir}")
        print("Abortando sem criar outputs/models/best_pose.pt.")
        return

    dest = Path("outputs/models/best_pose.pt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weight_src, dest)
    print(f"Peso encontrado: {weight_src}")
    print(f"Modelo copiado para: {dest}")

if __name__ == "__main__":
    main()
