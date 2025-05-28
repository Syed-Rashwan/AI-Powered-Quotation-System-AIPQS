# AI-Powered Quotation System (AIPQS)

An advanced, intelligent system designed to process architectural blueprints for precise detection of rooms, doors, and automated generation of electrical device quotations. Leveraging state-of-the-art YOLOv8 object detection and robust OCR technologies, AIPQS delivers unparalleled accuracy and efficiency in architectural analysis and quotation automation.

## Key Features

- **Accurate Room and Door Detection**: Utilizes the cutting-edge YOLOv8 model to identify rooms and doors within architectural blueprints with high precision.
- **Sophisticated Room Name Recognition**: Employs a hybrid OCR pipeline combining EasyOCR and pytesseract, enhanced with advanced preprocessing techniques for reliable text extraction.
- **Automated Quotation Generation**: Calculates electrical device requirements automatically based on detected room types, streamlining the estimation process.
- **Interactive and User-Friendly Interface**: Features a responsive web UI integrated with a chatbot assistant to guide users seamlessly through the workflow.

## Project Architecture

```
.
├── src/                    # Core source code modules
│   ├── app.py              # Flask web application entry point
│   ├── chatbot.py          # Chatbot logic and interaction handling
│   ├── object_detection.py # YOLOv8-based detection algorithms
│   ├── ocr_blueprint_pipeline.py # OCR processing pipeline
│   ├── quotation_generator.py    # Quotation calculation logic
│   └── embedding_merge      # Text similarity and embedding utilities
├── templates/              # HTML templates for web UI
├── models/                 # Pretrained and custom-trained models
├── scripts/                # Utility and helper scripts
├── YOLOv8/                 # YOLOv8 model files and configurations
```

## System Requirements

- Python 3.8 or higher
- YOLOv8 framework
- EasyOCR library
- OpenCV for image processing
- pytesseract OCR engine
- Flask web framework
- Additional dependencies listed in `requirements.txt`

## Installation Guide

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI-Powered-Quotation-System-AIPQS.git
cd AI-Powered-Quotation-System-AIPQS
```

2. Set up and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install all required dependencies:
```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR engine:
- **Windows**: Download and install from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- **Linux**: `sudo apt install tesseract-ocr`
- **MacOS**: `brew install tesseract`

## Usage Instructions

1. Launch the Flask web application:
```bash
python src/app.py
```

2. Open your preferred web browser and navigate to:
```
http://localhost:5000
```

3. Upload architectural blueprint images and follow the intuitive interface prompts to generate quotations.

## Development Notes

- The YOLOv8 model is trained on the comprehensive CubiCasa5K dataset for robust room and door detection.
- The OCR pipeline integrates EasyOCR and pytesseract with advanced image preprocessing to maximize text recognition accuracy.
- Room names are matched using a combination of fuzzy string matching and embedding-based similarity techniques.
- Quotation generation is rule-based, tailored to each room type for precise electrical device estimation.

## Contribution Guidelines

We welcome contributions! Please follow these steps:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/YourFeatureName`
3. Commit your changes with clear messages: `git commit -m "Add feature description"`
4. Push your branch: `git push origin feature/YourFeatureName`
5. Open a Pull Request for review.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- The CubiCasa5K dataset for providing extensive training data.
- YOLOv8 for its powerful object detection capabilities.
- EasyOCR and Tesseract for enabling advanced OCR functionalities.
