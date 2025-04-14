import os
import shutil
from sklearn.model_selection import train_test_split

def structure_dataset(source_images_dir, source_labels_dir, output_dir, val_ratio=0.2):
    """
    Organize dataset into YOLOv8 compatible format.

    Args:
        source_images_dir (str): Directory containing all images.
        source_labels_dir (str): Directory containing all label txt files.
        output_dir (str): Directory to create structured dataset.
        val_ratio (float): Fraction of data to use for validation.

    Creates:
        output_dir/images/train, output_dir/images/val
        output_dir/labels/train, output_dir/labels/val
        output_dir/data.yaml
    """
    os.makedirs(output_dir, exist_ok=True)
    images_train_dir = os.path.join(output_dir, 'images', 'train')
    images_val_dir = os.path.join(output_dir, 'images', 'val')
    labels_train_dir = os.path.join(output_dir, 'labels', 'train')
    labels_val_dir = os.path.join(output_dir, 'labels', 'val')

    for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        os.makedirs(d, exist_ok=True)

    image_files = [f for f in os.listdir(source_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    train_files, val_files = train_test_split(image_files, test_size=val_ratio, random_state=42)

    def copy_files(file_list, images_dest, labels_dest):
        for file_name in file_list:
            src_img = os.path.join(source_images_dir, file_name)
            src_label = os.path.join(source_labels_dir, os.path.splitext(file_name)[0] + '.txt')
            dst_img = os.path.join(images_dest, file_name)
            dst_label = os.path.join(labels_dest, os.path.splitext(file_name)[0] + '.txt')

            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
            else:
                print(f"Warning: Image file {src_img} not found.")

            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)
            else:
                print(f"Warning: Label file {src_label} not found.")

    copy_files(train_files, images_train_dir, labels_train_dir)
    copy_files(val_files, images_val_dir, labels_val_dir)

    # Create data.yaml
    data_yaml = f"""
train: {images_train_dir}
val: {images_val_dir}

nc: 1
names: ['object']
"""
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(data_yaml.strip())

    print(f"Dataset structured in {output_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Structure dataset in YOLOv8 format")
    parser.add_argument('source_images_dir', type=str, help='Directory with all images')
    parser.add_argument('source_labels_dir', type=str, help='Directory with all label txt files')
    parser.add_argument('output_dir', type=str, help='Output directory for structured dataset')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio')

    args = parser.parse_args()

    structure_dataset(args.source_images_dir, args.source_labels_dir, args.output_dir, args.val_ratio)
