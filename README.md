# Electrical Blueprint Quotation Generator

This project automates the process of analyzing electrical blueprints, detecting electrical components, generating quotations, and producing professional PDF reports.

## Features

- Dataset preparation and annotation for YOLO object detection.
- YOLOv8 model training on blueprint images.
- Object detection inference on uploaded blueprints.
- Quotation generation based on detected components.
- PDF report generation for quotations.
- Web UI for uploading blueprints and downloading reports.

## Installation

1. Clone the repository.

2. Create and activate a Python virtual environment (recommended).

3. Install dependencies:

```
pip install -r requirements.txt
```

## Usage

### Train the Model

```bash
python models/train_yolo.py
```

### Run Inference and Generate Quotation

```bash
python main.py path/to/blueprint_image.png
```

### Run the Web Application

```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000` to upload blueprints and download quotations.

### Test Detection

```bash
python test_detection.py
```

## Project Structure

- `datasets/` - Dataset and annotation scripts.
- `models/` - Model training scripts.
- `inference.py` - YOLO inference script.
- `object_detection.py` - Object detection script.
- `quotation_generator.py` - Quotation generation module.
- `quotation_generation.py` - Alternative quotation generation module.
- `report_generator.py` - PDF report generation module.
- `app.py` - Flask web application.
- `main.py` - Main pipeline script.
- `test_detection.py` - Detection test script.
- `requirements.txt` - Python dependencies.

## License

MIT License
