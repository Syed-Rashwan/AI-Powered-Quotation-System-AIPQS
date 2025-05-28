from ultralytics import YOLO
import cv2
import os
# from roboflow import Roboflow

def run_inference(image_path, model_path='models/yolov8n_trained.pt', conf_threshold=0.25):
    """
    Run YOLOv8 inference on the given image using the specified local model path.
    """
    # Load the local YOLO model
    model = YOLO(model_path)

    # Read the input image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
         
    # Run inference
    results = model(img)

    # Parse results
    detections = []
    for result in results:
        for box in result.boxes:
            conf = box.conf.item()
            if conf >= conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls.item())
                detections.append({
                    'class_id': class_id,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                })

    return detections

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLO inference on blueprint image")
    parser.add_argument('image_path', type=str, help='Path to blueprint image')
    parser.add_argument('--model_path', type=str, default='models/yolov8n_trained.pt', help='Local YOLO model path')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='Confidence threshold for detections')

    args = parser.parse_args()

    detections = run_inference(args.image_path, args.model_path, args.conf_threshold)
    print("Detections:")
    for det in detections:
        print(det)
