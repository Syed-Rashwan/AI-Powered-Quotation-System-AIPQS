import os
from cairosvg import svg2png

def convert_svg_folder_to_png(svg_folder, png_folder):
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)

    svg_files = [f for f in os.listdir(svg_folder) if f.lower().endswith('.svg')]
    if not svg_files:
        print(f"No SVG files found in {svg_folder}")
        return

    for svg_file in svg_files:
        svg_path = os.path.join(svg_folder, svg_file)
        png_file = os.path.splitext(svg_file)[0] + '.png'
        png_path = os.path.join(png_folder, png_file)
        try:
            svg2png(url=svg_path, write_to=png_path)
            print(f"Converted {svg_file} to {png_file}")
        except Exception as e:
            print(f"Failed to convert {svg_file}: {e}")

if __name__ == "__main__":
    # Example usage: convert all SVGs in datasets/raw/Train 1 to PNGs in datasets/raw/Train 1_png
    svg_folder = "datasets/raw/Train 1"
    png_folder = "datasets/raw/Train 1_png"
    convert_svg_folder_to_png(svg_folder, png_folder)
