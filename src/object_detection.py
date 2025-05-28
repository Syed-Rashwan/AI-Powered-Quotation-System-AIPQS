from ultralytics import YOLO
import cv2
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -- Detect objects --
def detect_objects(image_paths, model_path='models/yolov8m.pt', conf_threshold=0.25, class_filter=None, save_annotated=False, output_dir='output'):
    """
    Detect objects in one or multiple images using YOLO model.

    Args:
        image_paths (str or list): Path to image or list of image paths or directory containing images.
        model_path (str): Path to trained YOLO model.
        conf_threshold (float): Confidence threshold to filter detections.
        class_filter (list or None): List of class ids to keep. If None, keep all.
        save_annotated (bool): Whether to save annotated images with bounding boxes.
        output_dir (str): Directory to save annotated images if save_annotated is True.

    Returns:
        dict: Mapping image_path -> list of detections (dict with class_id, class_name, confidence, bbox)
    """
    # Load the trained YOLO model
    model = YOLO(model = YOLO('runs/detect/train7/weights/best.pt'))

    # Prepare list of image paths
    if isinstance(image_paths, str):
        if os.path.isdir(image_paths):
            exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_paths = [os.path.join(image_paths, f) for f in os.listdir(image_paths) if os.path.splitext(f)[1].lower() in exts]
        else:
            image_paths = [image_paths]
    elif not isinstance(image_paths, list):
        raise ValueError("image_paths must be a string or list of strings")

    if save_annotated and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results_dict = {}

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Image not found or cannot be read: {img_path}")
            continue

        results = model(img)

        detections = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < conf_threshold:
                    continue
                if class_filter is not None and cls not in class_filter:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = model.names.get(cls, f"class_{cls}")
                detections.append({
                    'class_id': cls,
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                })

        results_dict[img_path] = detections

        if save_annotated:
            annotated_img = results[0].plot()
            save_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(save_path, annotated_img)
            logger.info(f"Saved annotated image to {save_path}")

    return results_dict

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run object detection on images or directory")
    parser.add_argument('input_path', type=str, help='Path to image file or directory of images')
    parser.add_argument('--model_path', type=str, default='models/yolov8n_trained.pt', help='Path to trained YOLO model')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='Confidence threshold for detections')
    parser.add_argument('--class_filter', type=int, nargs='*', default=None, help='List of class IDs to filter')
    parser.add_argument('--save_annotated', action='store_true', help='Save annotated images with bounding boxes')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save annotated images')

    args = parser.parse_args()

    detections = detect_objects(
        args.input_path,
        model_path=args.model_path,
        conf_threshold=args.conf_threshold,
        class_filter=args.class_filter,
        save_annotated=args.save_annotated,
        output_dir=args.output_dir
    )

    for img_path, dets in detections.items():
        print(f"Detections for {img_path}:")
        for det in dets:
            print(det)
