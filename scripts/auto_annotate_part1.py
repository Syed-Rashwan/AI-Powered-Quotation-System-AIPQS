import cv2
import os
from pathlib import Path

def auto_annotate(image_path, output_path):
    # Load blueprint image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Basic processing to find contours
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Write annotations in YOLO format
    with open(output_path, 'w') as f:
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:  # Filter small elements
                x,y,w,h = cv2.boundingRect(cnt)
                # Normalize coordinates (YOLO format)
                x_center = (x + w/2) / img.shape[1]
                y_center = (y + h/2) / img.shape[0]
                width = w / img.shape[1]
                height = h / img.shape[0]
                # Class 0 as placeholder, can be refined later
                f.write(f'0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n')

def main():
    image_dir = Path('datasets/part_1/train')
    label_dir = Path('datasets/part_1/labels')
    label_dir.mkdir(exist_ok=True)
    
    for img_path in image_dir.glob('*.png'):
        label_path = label_dir / f'{img_path.stem}.txt'
        auto_annotate(str(img_path), str(label_path))
        print(f'Annotated {img_path.name} -> {label_path.name}')

if __name__ == "__main__":
    main()
