import os
from pathlib import Path

def check_yolo_dataset(dataset_dir, num_classes):
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / 'images'
    labels_dir = dataset_dir / 'labels'

    if not images_dir.exists() or not labels_dir.exists():
        print(f"Error: images or labels directory missing in {dataset_dir}")
        return False

    image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg'))
    label_files = list(labels_dir.glob('*.txt'))

    if len(image_files) == 0:
        print("No image files found.")
        return False
    if len(label_files) == 0:
        print("No label files found.")
        return False
    if len(image_files) != len(label_files):
        print(f"Warning: Number of images ({len(image_files)}) and labels ({len(label_files)}) do not match.")

    valid = True
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"Invalid label format in {label_file} line {i+1}")
                    valid = False
                    continue
                class_id, x_center, y_center, width, height = parts
                try:
                    class_id = int(class_id)
                    x_center = float(x_center)
                    y_center = float(y_center)
                    width = float(width)
                    height = float(height)
                except ValueError:
                    print(f"Invalid number format in {label_file} line {i+1}")
                    valid = False
                    continue
                if not (0 <= class_id < num_classes):
                    print(f"Class id {class_id} out of range in {label_file} line {i+1}")
                    valid = False
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                    print(f"Bounding box values out of range in {label_file} line {i+1}")
                    valid = False
    if valid:
        print(f"Dataset at {dataset_dir} looks valid for training.")
    else:
        print(f"Dataset at {dataset_dir} has errors.")
    return valid

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python check_dataset.py <dataset_dir> <num_classes>")
        sys.exit(1)
    dataset_dir = sys.argv[1]
    num_classes = int(sys.argv[2])
    check_yolo_dataset(dataset_dir, num_classes)
