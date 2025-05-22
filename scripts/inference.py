from ultralytics import YOLO
import cv2
import os
from roboflow import Roboflow

def run_inference(image_path, model_id='builderformer/10', conf_threshold=0.25):
    """
    Run YOLOv8 inference on the given image using the specified Roboflow model ID.
    """
    # Initialize Roboflow and load the model
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    workspace_name, version_str = model_id.split('/')
    project = rf.workspace(workspace_name).project(workspace_name)
    model = project.version(int(version_str)).model

    # Read the input image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Run inference
    results = model.predict(img).json()

    # Parse results
    detections = []
    for prediction in results.get('predictions', []):
        conf = prediction.get('confidence', 0)
        if conf >= conf_threshold:
            bbox = prediction.get('bbox', {})
            x1 = int(bbox.get('x', 0) - bbox.get('width', 0) / 2)
            y1 = int(bbox.get('y', 0) - bbox.get('height', 0) / 2)
            x2 = int(bbox.get('x', 0) + bbox.get('width', 0) / 2)
            y2 = int(bbox.get('y', 0) + bbox.get('height', 0) / 2)
            detections.append({
                'class_id': prediction.get('class', ''),
                'confidence': conf,
                'bbox': (x1, y1, x2, y2)
            })

    return detections

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLO inference on blueprint image")
    parser.add_argument('image_path', type=str, help='Path to blueprint image')
    parser.add_argument('--model_id', type=str, default='builderformer/10', help='Roboflow model ID')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='Confidence threshold for detections')

    args = parser.parse_args()

    detections = run_inference(args.image_path, args.model_id, args.conf_threshold)
    print("Detections:")
    for det in detections:
        print(det)