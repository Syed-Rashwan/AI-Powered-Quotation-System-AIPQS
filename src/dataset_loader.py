import os
import xml.etree.ElementTree as ET
from PIL import Image

class CubiCasaDataset:
    def __init__(self, base_path, split_file):
        self.base_path = base_path
        self.split_file = split_file
        self.samples = self._load_split()

    def _load_split(self):
        samples = []
        split_path = os.path.join(self.base_path, self.split_file)
        with open(split_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Remove leading slash if present
                    if line.startswith('/'):
                        line = line[1:]
                    samples.append(line)
        return samples

    def load_image(self, sample_folder):
        img_path = os.path.join(self.base_path, sample_folder, 'F1_original.png')
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        return Image.open(img_path)

    def parse_svg(self, sample_folder):
        svg_path = os.path.join(self.base_path, sample_folder, 'model.svg')
        if not os.path.exists(svg_path):
            raise FileNotFoundError(f"SVG file not found: {svg_path}")
        tree = ET.parse(svg_path)
        root = tree.getroot()
        ns = {'svg': 'http://www.w3.org/2000/svg'}

        # Extract polygons whose parent <g> has class starting with "Space"
        polygons = []
        for polygon in root.findall('.//svg:polygon', ns):
            parent = polygon.getparent() if hasattr(polygon, 'getparent') else None
            # xml.etree.ElementTree does not support getparent, so we find parent manually
            # We will iterate through all <g> elements and check their children
            # So instead, we will do a two-pass approach:
            # First, find all <g> elements with class starting with "Space"
            # Then find polygons inside them
            # So we break here and implement a different approach below
            pass

        # New approach: find all <g> elements with class starting with "Space"
        polygons = []
        for g in root.findall('.//svg:g', ns):
            class_attr = g.attrib.get('class', '')
            if class_attr.startswith('Space'):
                for polygon in g.findall('svg:polygon', ns):
                    points = polygon.attrib.get('points', '')
                    points_list = []
                    for point in points.strip().split():
                        x_str, y_str = point.split(',')
                        points_list.append((float(x_str), float(y_str)))
                    # Extract room name from <text> element with class "NameLabel"
                    room_name = None
                    for text_g in g.findall('svg:g', ns):
                        if text_g.attrib.get('class', '') == 'TextLabel NameLabel':
                            text_elem = text_g.find('svg:text', ns)
                            if text_elem is not None and text_elem.text:
                                room_name = text_elem.text.strip()
                                break
                    print(f"DEBUG: Found polygon with room_name: {room_name}")
                    polygons.append({
                        'class': class_attr,
                        'room_name': room_name,
                        'points': points_list
                    })
        return polygons

if __name__ == '__main__':
    # Example usage and test
    base_path = os.path.join(os.path.dirname(__file__), '..', 'cubicasa5k', 'cubicasa5k')
    train_split = 'train.txt'
    dataset = CubiCasaDataset(base_path, train_split)
    print(f"Number of training samples: {len(dataset.samples)}")
    sample_folder = dataset.samples[0]
    print(f"Loading sample: {sample_folder}")
    img = dataset.load_image(sample_folder)
    print(f"Image size: {img.size}")
    polygons = dataset.parse_svg(sample_folder)
    print(f"Number of polygons in SVG: {len(polygons)}")
    for poly in polygons[:3]:
        print(f"Class: {poly['class']}, Points: {poly['points'][:3]} ...")
