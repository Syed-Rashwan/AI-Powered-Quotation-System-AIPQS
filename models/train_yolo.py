from ultralytics import YOLO
import os

def train_yolo_model(dataset_path="Dataset", model_save_path="models/yolov8m.pt", epochs=100, 
                     batch_size=16, imgsz=640, lr=0.01, fast_train=False, pretrained_model_path="models/yolov8m.pt", resume_training=False):
    # Verify dataset exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    # Load pretrained YOLOv8 model or resume from trained model
    model = YOLO(pretrained_model_path)  # Load model

    # Check for GPU availability and use GPU if available
    import torch
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {'GPU' if device == '0' else 'CPU'}")

    # Compose absolute path to data.yaml
    data_yaml_path = os.path.abspath(os.path.join(dataset_path, "data.yaml"))

    # Adjust parameters for faster training if requested
    if fast_train:
        epochs = min(epochs, 10)
        batch_size = min(batch_size, 8)
        imgsz = min(imgsz, 320)
        print("Fast training mode enabled: reduced epochs, batch size, and image size.")

    # Train the model with data augmentation (enabled by default in YOLOv8)
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        lr0=lr,
        workers=4 if device == '0' else 2,
        save=True,  # Save best model automatically
        #patience=10,  # Early stopping patience
        resume=False  # Always start new training, do not resume
    )

    # Save the trained model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    # Example: continue training from trained model for 80 more epochs without resume flag
    train_yolo_model(
        dataset_path="Dataset",
        model_save_path="models/yolov8m.pt",
        epochs=80,
        pretrained_model_path="models/yolov8m.pt",
        resume_training=False
    )
