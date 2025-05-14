import os

def clean_dataset(base_path):
    """
    Deletes 'original' and 'F2_scaled' images in numbered folders under base_path,
    keeping only 'F1_scaled' images and SVG files.
    """
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path) and folder_name.isdigit():
            print(f"Processing folder: {folder_path}")
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    # Delete files that are original or F2_scaled images
                    if filename in ['F1_original.png', 'F2_original.png', 'F2_scaled.png']:
                        print(f"Deleting file: {file_path}")
                        os.remove(file_path)

if __name__ == "__main__":
    base_dataset_path = "cubicasa5k/cubicasa5k/high_quality_architectural"
    clean_dataset(base_dataset_path)
