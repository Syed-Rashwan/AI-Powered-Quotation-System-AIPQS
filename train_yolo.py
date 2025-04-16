from ultralytics import YOLO
import os

def get_latest_model_path(models_dir="models", base_model_name="yolov8m_trained.pt"):
    # Find the latest model file based on modification time
    model_files = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.startswith("yolov8m_trained")
                    and f.endswith(".pt")]
    if not model_files:
        return os.path.join(models_dir, base_model_name)
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

def train_yolo_model(dataset_path="datasets/part_1", model_save_path="models/yolov8m_trained.pt", 
                     epochs=20, batch_size=16, imgsz=640, lr=0.01, fast_train=False):
    # Verify dataset exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    # Load latest pretrained YOLOv8 medium model for better accuracy
    pretrained_model_path = get_latest_model_path()
    print(f"Using pretrained model: {pretrained_model_path}")
    model = YOLO(pretrained_model_path)  # Load pretrained model

    # Check for GPU availability and use GPU if available
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {'GPU' if device == 'cuda' else 'CPU'}")

    # Compose absolute path to data.yaml
    data_yaml_path = os.path.abspath(os.path.join(dataset_path, "data.yaml"))

    # Adjust parameters for faster training if requested
    if fast_train:
        epochs = min(epochs, 10)
        batch_size = min(batch_size, 8)
        imgsz = min(imgsz, 320)
        print("Fast training mode enabled: reduced epochs, batch size, and image size.")

    # Disable resume training to avoid assertion error when starting fresh
    resume_training = False

    # Train the model with data augmentation (enabled by default in YOLOv8)
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        lr0=lr,
        workers=4 if device == '0' else 2,
        save=True,  # Saves best model automatically
        patience=10,  # Early stopping patience 
        resume=resume_training,  # Do not resume training to avoid errors
        amp=True,  # Enable mixed precision training for speedup
        cache=False  # Disable data caching to avoid cache issues
    )

    # Save the trained model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_yolo_model()
