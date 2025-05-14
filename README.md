# AI Powered Quotation System (AIPQS)

This project automates the process of analyzing electrical blueprints, detecting electrical components, generating quotations, and producing professional PDF reports.

## Features

- Dataset preparation and annotation for YOLO object detection.
- YOLOv8 model training on blueprint images.
- Object detection inference on uploaded blueprints.
- Quotation generation based on detected components.
- PDF report generation for quotations.
- Web UI for uploading blueprints and downloading reports.
- Containerized deployment with Docker for easy setup.

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
python src/app.py
```

Open your browser at `http://127.0.0.1:5000` to upload blueprints and download quotations.

### Run with Docker

Build the Docker image:

```bash
docker build -t aipqs .
```

Run the Docker container:

```bash
docker run -p 5000:5000 -e FLASK_SECRET_KEY=your_secret_key aipqs
```

### Test Detection

```bash
python test_detection.py
```

## Project Structure

```plaintext
AIPQS/
├── datasets/                # Dataset and annotation scripts
├── models/                 # Model training scripts
├── scripts/                # Inference and dataset preparation scripts
├── src/                    # Source code for app, detection, quotation, report
├── templates/              # HTML templates for web UI
├── runs/                   # Training and inference output
├── app.py                  # Flask web application entry point (moved to src/)
├── main.py                 # Main pipeline script
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker container configuration
└── README.md               # Project documentation
```

## License

This project is under MIT License.

## Contributors
- Syed Rashwan (Project lead)
