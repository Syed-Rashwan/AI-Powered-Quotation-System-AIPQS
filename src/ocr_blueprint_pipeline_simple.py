import cv2
import json
import numpy as np
from collections import Counter
from tabulate import tabulate
from PIL import Image
import embedding_merge
from embedding_merge import EmbeddingMerger
from difflib import get_close_matches
import sys
import os
import easyocr
import pytesseract


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Define the fixed rule mapping for each room type
ROOM_DEVICE_RULES = {
    "Bedroom": {"light": 3, "switch": 4, "thermostat": 1},
    "Kitchen": {"light": 2, "switch": 3, "chimney": 1},
    "Bathroom": {"light": 1, "switch": 1, "exhaust fan": 1},
    "Living Room": {"light": 4, "switch": 5, "thermostat": 1},
    "Dining Room": {"light": 2, "switch": 3},
}

# List of known room types for filtering OCR results
KNOWN_ROOMS = set(ROOM_DEVICE_RULES.keys())

# Synonyms dictionary 
ROOM_SYNONYMS = {
    "Bed Room": "Bedroom",
    "Master Bedroom": "Bedroom",
    "Bed": "Bedroom",
    "Bedroom 2": "Bedroom",
    "Kitchen Room": "Kitchen",
    "Bath Room": "Bathroom",
    "Living": "Living Room",
    "Dinning Room": "Dining Room",
    "Bath room": "Bathroom",
    "Master bath": "Bathroom",
    "Bath": "Bathroom",
    "WC": "Bathroom",
}
# -- Detect rooms --
def detect_rooms(image_path, show_image=False):
    """
    Detect room labels in the blueprint image using EasyOCR and pytesseract with simple preprocessing.
    Uses fuzzy matching to allow minor OCR errors in room names and synonyms.
    Combines results from both OCR engines to improve recall.
    Uses embedding-based merging to reduce duplicate detections.
    Returns detected room labels and their bounding boxes.
    """
    import easyocr
    import pytesseract
    import numpy as np

    # Initialize EasyOCR reader for English with GPU enabled
    reader = easyocr.Reader(['en'], gpu=True)

    # Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found or unable to read: {image_path}")

    # Simple preprocessing: convert to grayscale and apply median blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("debug_gray.png", gray)

    blurred = cv2.medianBlur(gray, 3)
    cv2.imwrite("debug_blurred.png", blurred)

    preprocessed_image = blurred
    cv2.imwrite("debug_preprocessed.png", preprocessed_image)

    # Convert to BGR for EasyOCR compatibility
    preprocessed_bgr = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)

    # Perform OCR detection with EasyOCR with adjusted parameters
    easyocr_results = reader.readtext(
        preprocessed_bgr,
        contrast_ths=0.1,
        adjust_contrast=0.8,
        text_threshold=0.2,
        low_text=0.2,
        width_ths=0.5,
        paragraph=False
    )

    # Perform OCR detection with pytesseract
    pytesseract_text = pytesseract.image_to_string(preprocessed_image, lang='eng', config='--psm 6')
    pytesseract_lines = [line.strip() for line in pytesseract_text.split('\n') if line.strip()]

    detected_rooms = []
    bounding_boxes = []
    confidences = []

    # Process EasyOCR results
    for bbox, text, confidence in easyocr_results:
        normalized_text = text.strip().title()
        if normalized_text in ROOM_SYNONYMS:
            matched_room = ROOM_SYNONYMS[normalized_text]
        else:
            matches = get_close_matches(normalized_text, KNOWN_ROOMS, n=1, cutoff=0.7)
            matched_room = matches[0] if matches else None
        if matched_room and confidence > 0.5:
            detected_rooms.append(matched_room)
            bounding_boxes.append(bbox)
            confidences.append(confidence)

    # Process pytesseract results (no bounding boxes available)
    for line in pytesseract_lines:
        normalized_text = line.title()
        if normalized_text in ROOM_SYNONYMS:
            matched_room = ROOM_SYNONYMS[normalized_text]
        else:
            matches = get_close_matches(normalized_text, KNOWN_ROOMS, n=1, cutoff=0.6)
            matched_room = matches[0] if matches else None
        if matched_room:
            detected_rooms.append(matched_room)
            bounding_boxes.append(None)
            confidences.append(0)

    # Merge detected room labels using embedding-based merging to reduce duplicates
    merger = EmbeddingMerger(similarity_threshold=0.95)
    merged_rooms, cluster_indices = merger.merge_similar_texts_with_indices(detected_rooms)

    # For each cluster, pick the bounding box with highest confidence
    merged_bounding_boxes = []
    for indices in cluster_indices:
        cluster_bboxes = [bounding_boxes[i] for i in indices if bounding_boxes[i] is not None]
        cluster_confidences = [confidences[i] for i in indices if bounding_boxes[i] is not None]
        if cluster_bboxes:
            max_idx = cluster_confidences.index(max(cluster_confidences))
            merged_bounding_boxes.append(cluster_bboxes[max_idx])
        else:
            merged_bounding_boxes.append(None)

    # Debug: print merged detected rooms
    print("\nMerged Detected Rooms after Embedding-based Merging:")
    for room in merged_rooms:
        print(f"Room: {room}")

    # Display detected rooms on image with bounding boxes if available
    draw_bounding_boxes(image, merged_bounding_boxes, merged_rooms)

    # Save the image with bounding boxes instead of displaying it
    output_path = "detected_room_labels_output.png"
    cv2.imwrite(output_path, image)
    print(f"Output image with detected room labels saved to {output_path}")

    return merged_rooms, merged_bounding_boxes, image

#-- Count devices -- 
def count_devices(room_counts):
    """
    Given a dictionary of room counts, calculate total devices needed.
    """
    total_devices = Counter()
    for room, count in room_counts.items():
        devices = ROOM_DEVICE_RULES.get(room, {})
        for device, qty in devices.items():
            total_devices[device] += qty * count
    return dict(total_devices)

def draw_bounding_boxes(image, bounding_boxes, labels):
    """
    Draw bounding boxes and labels on the image using OpenCV.
    """
    for bbox, label in zip(bounding_boxes, labels):
        if bbox is None:
            continue
        pts = np.array(bbox, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        top_left = tuple(pts[0][0])
        cv2.putText(image, label, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Default sample image path
SAMPLE_IMAGE_PATH = "C:/Users/Rashwan Syed/Desktop/AIPQS/sample_image_enhanced.png"

def main(image_path=SAMPLE_IMAGE_PATH, export_json=False, json_path="device_counts.json"):
    """
    Main function to run the OCR pipeline on the blueprint image.
    """
    detected_rooms, bounding_boxes, image = detect_rooms(image_path)
    room_counts = Counter(detected_rooms)
    total_devices = count_devices(room_counts)

    print("\nRoom Counts:")
    print(tabulate(room_counts.items(), headers=["Room Type", "Count"], tablefmt="grid"))

    print("\nTotal Devices Needed:")
    print(tabulate(total_devices.items(), headers=["Device", "Quantity"], tablefmt="grid"))

    draw_bounding_boxes(image, bounding_boxes, detected_rooms)

    cv2.imshow("Detected Room Labels", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if export_json:
        output = {
            "room_counts": dict(room_counts),
            "total_devices": total_devices
        }
        with open(json_path, "w") as f:
            json.dump(output, f, indent=4)
        print(f"\nResults exported to {json_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OCR pipeline for blueprint room detection and device counting.")
    parser.add_argument("image_path", nargs='?', default=None, help="Path to the blueprint image (JPEG or PNG).")
    parser.add_argument("--export-json", action="store_true", help="Export results to a JSON file.")
    parser.add_argument("--json-path", default="device_counts.json", help="Path for the JSON output file.")

    args = parser.parse_args()

    if args.image_path is None:
        main(export_json=args.export_json, json_path=args.json_path)
    else:
        main(args.image_path, export_json=args.export_json, json_path=args.json_path)
