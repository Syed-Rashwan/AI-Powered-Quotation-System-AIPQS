from ultralytics import YOLO
import cv2
import os

def run_inference(image_path, model_path='models/yolov8m_trained.pt', conf_threshold=0.25):
    # Load the trained YOLO model
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
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf >= conf_threshold:
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    'class_id': cls,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                })

    return detections

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLO inference on blueprint image")
    parser.add_argument('image_path', type=str, help='Path to blueprint image')
    parser.add_argument('--model_path', type=str, default='models/yolov8n_fifth_epoch.pt', help='Path to trained YOLO model')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='Confidence threshold for detections')

    args = parser.parse_args()

    detections = run_inference(args.image_path, args.model_path, args.conf_threshold)
    print("Detections:")
    for det in detections:
        print(det)
