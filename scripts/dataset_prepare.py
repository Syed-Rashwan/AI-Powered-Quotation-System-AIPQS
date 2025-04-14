import os
import json
import shutil
from sklearn.model_selection import train_test_split

def convert_bbox_to_yolo(size, bbox):
    """
    Convert COCO bbox [x_min, y_min, width, height] to YOLO format [x_center, y_center, width, height]
    All values normalized by image width and height.
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = bbox[0] + bbox[2] / 2.0
    y = bbox[1] + bbox[3] / 2.0
    w = bbox[2]
    h = bbox[3]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def prepare_dataset(coco_annotation_file, images_dir, output_dir, val_ratio=0.2):
    """
    Convert COCO annotations to YOLO format and organize dataset.

    Args:
        coco_annotation_file (str): Path to COCO JSON annotation file.
        images_dir (str): Directory containing images.
        output_dir (str): Directory to save YOLO formatted dataset.
        val_ratio (float): Ratio of validation set.

    Output:
        Creates train/val folders with images and labels.
        Creates data.yaml file.
    """
    with open(coco_annotation_file, 'r') as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco['images']}
    annotations = coco['annotations']
    categories = {cat['id']: cat['name'] for cat in coco['categories']}
    class_to_id = {name: idx for idx, name in enumerate(sorted(categories.values()))}

    # Create output directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # Split images into train and val
    image_ids = list(images.keys())
    train_ids, val_ids = train_test_split(image_ids, test_size=val_ratio, random_state=42)

    # Map image_id to annotations
    img_to_anns = {}
    for ann in annotations:
        img_to_anns.setdefault(ann['image_id'], []).append(ann)

    def process_split(split_ids, split_name):
        for img_id in split_ids:
            img_info = images[img_id]
            img_filename = img_info['file_name']
            img_path = os.path.join(images_dir, img_filename)
            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} not found, skipping.")
                continue

            # Copy image
            dst_img_path = os.path.join(output_dir, 'images', split_name, img_filename)
            shutil.copy(img_path, dst_img_path)

            # Create label file
            label_filename = os.path.splitext(img_filename)[0] + '.txt'
            label_path = os.path.join(output_dir, 'labels', split_name, label_filename)

            anns = img_to_anns.get(img_id, [])
            with open(label_path, 'w') as lf:
                for ann in anns:
                    class_name = categories[ann['category_id']]
                    class_id = class_to_id[class_name]
                    bbox = ann['bbox']
                    size = (img_info['width'], img_info['height'])
                    x, y, w, h = convert_bbox_to_yolo(size, bbox)
                    lf.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    process_split(train_ids, 'train')
    process_split(val_ids, 'val')

    # Create data.yaml
    data_yaml = f"""
train: {os.path.join(output_dir, 'images', 'train')}
val: {os.path.join(output_dir, 'images', 'val')}

nc: {len(class_to_id)}
names: {list(class_to_id.keys())}
"""
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(data_yaml.strip())

    print(f"Dataset prepared in {output_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from COCO annotations")
    parser.add_argument('coco_annotation_file', type=str, help='Path to COCO JSON annotation file')
    parser.add_argument('images_dir', type=str, help='Directory containing images')
    parser.add_argument('output_dir', type=str, help='Output directory for YOLO dataset')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio')

    args = parser.parse_args()

    prepare_dataset(args.coco_annotation_file, args.images_dir, args.output_dir, args.val_ratio)
