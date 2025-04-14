import os
from PIL import Image, ImageDraw

def simple_svg_to_png(svg_folder, png_folder):
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
            # This is a placeholder for a minimal SVG to PNG conversion.
            # Since full SVG parsing is complex, here we create a blank image as a stub.
            # You can extend this function to parse and draw SVG elements as needed.
            img = Image.new('RGB', (512, 512), color='white')
            draw = ImageDraw.Draw(img)
            # Example: draw a simple rectangle (replace with actual SVG parsing logic)
            draw.rectangle([50, 50, 462, 462], outline='black', width=5)
            img.save(png_path)
            print(f"Converted {svg_file} to {png_file} (placeholder image)")
        except Exception as e:
            print(f"Failed to convert {svg_file}: {e}")

if __name__ == "__main__":
    svg_folder = "datasets/raw/Train 1"
    png_folder = "datasets/raw/Train 1_png"
    simple_svg_to_png(svg_folder, png_folder)
