import os
import shutil
from pathlib import Path
from dataset_loader import CubiCasaDataset

def polygon_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return xmin, ymin, xmax, ymax

def convert_to_yolo_bbox(bbox, img_width, img_height):
    xmin, ymin, xmax, ymax = bbox
    x_center = (xmin + xmax) / 2.0 / img_width
    y_center = (ymin + ymax) / 2.0 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return x_center, y_center, width, height

def prepare_yolo_dataset(base_path, split_file, output_dir, class_map):
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    dataset = CubiCasaDataset(base_path, split_file)
    output_dir = Path(output_dir).resolve()
    logger.debug(f"Output directory resolved to: {output_dir}")
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created images directory: {images_dir}")
    logger.debug(f"Created labels directory: {labels_dir}")

    for sample_folder in dataset.samples:
        logger.debug(f"Processing sample: {sample_folder}")
        # Load image
        img = dataset.load_image(sample_folder)
        img_width, img_height = img.size

        # Copy image to output_dir/images
        src_img_path = os.path.join(base_path, sample_folder, 'F1_original.png')
        dst_img_path = images_dir / (Path(sample_folder).name + '.png')
        shutil.copy(src_img_path, dst_img_path)
        logger.debug(f"Copied image from {src_img_path} to {dst_img_path}")

        # Parse polygons
        polygons = dataset.parse_svg(sample_folder)

        # Create label file
        label_path = labels_dir / (Path(sample_folder).name + '.txt')
        with open(label_path, 'w') as f:
            for poly in polygons:
                room_name = poly.get('room_name')
                if room_name is None:
                    continue
                class_id = class_map.get(room_name.lower())
                if class_id is None:
                    # Skip unknown classes
                    continue
                bbox = polygon_to_bbox(poly['points'])
                yolo_bbox = convert_to_yolo_bbox(bbox, img_width, img_height)
                x_center, y_center, width, height = yolo_bbox
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        logger.debug(f"Wrote label file: {label_path}")

if __name__ == '__main__':
    # Example usage
    base_path = os.path.join(os.path.dirname(__file__), '..', 'cubicasa5k', 'cubicasa5k')
    split_file = 'train.txt'
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'yolo_dataset', 'train')

    # Define room classes and their IDs (example)
    class_map = {
        'bathroom': 0,
        'kitchen': 1,
        'bedroom': 2,
        'living room': 3,
        'terrace': 4,
        'dining room': 5,
        'balcony': 6,
        'hallway': 7,
        # aliases
        'bath': 0,
        'livingroom': 3,
        'diningroom': 5,
        'outdoor terrace': 4,
    }

    prepare_yolo_dataset(base_path, split_file, output_dir, class_map)
