# AI-Powered Quotation System (AIPQS)

An intelligent system that processes architectural blueprints to detect rooms, doors, and generate electrical device quotations using YOLOv8 object detection and OCR.

## Features

- **Room and Door Detection**: Uses YOLOv8 model for detecting rooms and doors in architectural blueprints
- **Room Name Recognition**: OCR pipeline using EasyOCR and pytesseract for detecting room names
- **Quotation Generation**: Automatically calculates required electrical devices based on room types
- **Interactive UI**: Web interface with chatbot assistance for seamless user experience

## Project Structure

```
.
├── src/                    # Source code
│   ├── app.py             # Flask web application
│   ├── chatbot.py         # Chatbot implementation
│   ├── object_detection.py # YOLOv8 detection logic
│   ├── ocr_blueprint_pipeline.py # OCR pipeline
│   ├── quotation_generator.py # Quotation generation
│   └── embedding_merge    # Text similarity module
├── templates/             # HTML templates
├── models/               # Trained models
├── scripts/             # Utility scripts
├── YOLOv8/             # YOLOv8 model files
└── cubicasa5k/         # Dataset files
```

## Requirements

- Python 3.8+
- YOLOv8
- EasyOCR
- OpenCV
- pytesseract
- Flask
- Other dependencies in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI-Powered-Quotation-System-AIPQS.git
cd AI-Powered-Quotation-System-AIPQS
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR:
- Windows: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- Linux: `sudo apt install tesseract-ocr`
- Mac: `brew install tesseract`

## Usage

1. Start the Flask application:
```bash
python src/app.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

3. Upload a blueprint image and follow the interface instructions.

## Development

- The YOLOv8 model is trained on the CubiCasa5K dataset for room and door detection
- OCR pipeline uses both EasyOCR and pytesseract with advanced preprocessing
- Room names are matched using fuzzy matching and embedding-based similarity
- Quotations are generated based on predefined rules for each room type

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CubiCasa5K dataset for training data
- YOLOv8 for object detection
- EasyOCR and Tesseract for OCR capabilities
