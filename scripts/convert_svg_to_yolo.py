import os
import xml.etree.ElementTree as ET
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def polygon_to_bbox(points):
    xs = []
    ys = []
    for p in points:
        if ',' in p:
            coords = p.split(',')
            if len(coords) == 2:
                x_str, y_str = coords
                try:
                    xs.append(float(x_str))
                    ys.append(float(y_str))
                except ValueError:
                    logger.warning(f"Invalid coordinate value: {p}")
                    continue
    if not xs or not ys:
        return 0, 0, 0, 0
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return xmin, ymin, xmax, ymax

def normalize_bbox(bbox, img_width, img_height):
    xmin, ymin, xmax, ymax = bbox
    x_center = (xmin + xmax) / 2 / img_width
    y_center = (ymin + ymax) / 2 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return x_center, y_center, width, height

def parse_svg(svg_path):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    annotations = []
    # Get image width and height from svg attributes
    img_width = float(root.attrib.get('width', '1'))
    img_height = float(root.attrib.get('height', '1'))

    for elem in root.iter():
        if elem.tag.endswith('polygon'):
            points_str = elem.attrib.get('points', '')
            points = points_str.strip().split(' ')
            if not points:
                continue
            # Get class label from multiple possible attributes
            label = None
            # Try common attributes for label
            for attr in ['class', 'label', 'name', 'id']:
                if attr in elem.attrib:
                    label = elem.attrib[attr]
                    break
            # If label still None, try parent's class attribute if possible
            if label is None:
                parent = elem.getparent() if hasattr(elem, 'getparent') else None
                if parent is not None and 'class' in parent.attrib:
                    label = parent.attrib['class']
            if label is None:
                label = 'unknown'
                logger.warning(f"Label not found for polygon in {svg_path}, defaulting to 'unknown'")
            annotations.append((label, points))
    return annotations, img_width, img_height

def main():
    # Define dataset root folder
    dataset_root = Path('cubicasa5k/cubicasa5k/high_quality_architectural')
    # Define output folder for YOLO annotations
    yolo_ann_root = Path('cubicasa5k/yolo_annotations')
    yolo_ann_root.mkdir(parents=True, exist_ok=True)

    # First pass: collect all unique class names
    class_names = set()
    for folder in dataset_root.glob('*'):
        if not folder.is_dir():
            continue
        svg_file = folder / 'model.svg'
        if not svg_file.exists():
            continue
        annotations, _, _ = parse_svg(svg_file)
        for label, _ in annotations:
            class_names.add(label)
    class_names = sorted(class_names)
    class_name_to_id = {name: idx for idx, name in enumerate(class_names)}

    # Second pass: process annotations and write YOLO files
    for folder in dataset_root.glob('*'):
        if not folder.is_dir():
            continue
        svg_file = folder / 'model.svg'
        img_file = None
        for f in folder.iterdir():
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                img_file = f
                break
        if not svg_file.exists() or not img_file:
            continue

        annotations, img_width, img_height = parse_svg(svg_file)
        ann_file = yolo_ann_root / (img_file.stem + '.txt')
        with open(ann_file, 'w') as f:
            for label, points in annotations:
                bbox = polygon_to_bbox(points)
                x_center, y_center, width, height = normalize_bbox(bbox, img_width, img_height)
                class_id = class_name_to_id.get(label, -1)
                if class_id == -1:
                    logger.warning(f"Class ID not found for label '{label}' in file {svg_file}")
                    continue
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # Write class names to file
    with open(yolo_ann_root / 'classes.txt', 'w') as f:
        for c in class_names:
            f.write(c + '\n')

    logger.info("Conversion complete. Classes saved to classes.txt")

if __name__ == '__main__':
    main()
