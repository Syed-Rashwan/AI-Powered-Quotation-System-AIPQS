import os
import shutil
import random

def split_dataset_into_parts(dataset_dir, parts=5, output_base_dir=None, copy=True, seed=42):
    """
    Split dataset images and labels into equal parts.

    Args:
        dataset_dir (str): Path to dataset directory containing 'train' and 'labels' subdirectories.
        parts (int): Number of parts to split into.
        output_base_dir (str or None): Base directory to save parts. If None, uses dataset_dir.
        copy (bool): If True, copy files; if False, move files.
        seed (int): Random seed for reproducibility.

    Returns:
        None
    """
    random.seed(seed)

    images_dir = os.path.join(dataset_dir, 'train')
    labels_dir = os.path.join(dataset_dir, 'labels')

    if output_base_dir is None:
        output_base_dir = dataset_dir

    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    image_files.sort()
    random.shuffle(image_files)

    total = len(image_files)
    part_size = total // parts
    remainder = total % parts

    start_idx = 0
    for i in range(parts):
        end_idx = start_idx + part_size + (1 if i < remainder else 0)
        part_files = image_files[start_idx:end_idx]

        part_dir = os.path.join(output_base_dir, f'training_data_{i+1}')
        part_images_dir = os.path.join(part_dir, 'train')
        part_labels_dir = os.path.join(part_dir, 'labels')

        os.makedirs(part_images_dir, exist_ok=True)
        os.makedirs(part_labels_dir, exist_ok=True)

        for img_file in part_files:
            label_file = os.path.splitext(img_file)[0] + '.txt'

            src_img_path = os.path.join(images_dir, img_file)
            src_label_path = os.path.join(labels_dir, label_file)

            dst_img_path = os.path.join(part_images_dir, img_file)
            dst_label_path = os.path.join(part_labels_dir, label_file)

            if copy:
                shutil.copy2(src_img_path, dst_img_path)
                if os.path.exists(src_label_path):
                    shutil.copy2(src_label_path, dst_label_path)
            else:
                shutil.move(src_img_path, dst_img_path)
                if os.path.exists(src_label_path):
                    shutil.move(src_label_path, dst_label_path)

        print(f"Part {i+1}: {len(part_files)} images")

        start_idx = end_idx

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split YOLO dataset into multiple parts")
    parser.add_argument('dataset_dir', type=str, help='Path to dataset directory containing train/ and labels/')
    parser.add_argument('--parts', type=int, default=5, help='Number of parts to split into')
    parser.add_argument('--output_base_dir', type=str, default=None, help='Base directory to save parts')
    parser.add_argument('--copy', action='store_true', help='Copy files instead of moving')
    parser.add_argument('--move', action='store_true', help='Move files instead of copying (default)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling')

    args = parser.parse_args()

    copy_files = True
    if args.move:
        copy_files = False
    elif args.copy:
        copy_files = True

    split_dataset_into_parts(
        args.dataset_dir,
        parts=args.parts,
        output_base_dir=args.output_base_dir,
        copy=copy_files,
        seed=args.seed
    )
