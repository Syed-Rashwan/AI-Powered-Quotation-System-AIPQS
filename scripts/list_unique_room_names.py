import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset_loader import CubiCasaDataset

def list_unique_room_names(base_path, split_file):
    dataset = CubiCasaDataset(base_path, split_file)
    unique_rooms = set()
    for sample_folder in dataset.samples:
        polygons = dataset.parse_svg(sample_folder)
        for poly in polygons:
            class_name = poly.get('class')
            if class_name and class_name.startswith('Space '):
                class_name_clean = class_name.replace('Space ', '').lower()
                unique_rooms.add(class_name_clean)
    return unique_rooms

if __name__ == '__main__':
    base_path = os.path.join(os.path.dirname(__file__), '..', 'cubicasa5k', 'cubicasa5k')
    split_file = 'train.txt'
    unique_rooms = list_unique_room_names(base_path, split_file)
    print("Unique room names in dataset:")
    for room in sorted(unique_rooms):
        print(room)
