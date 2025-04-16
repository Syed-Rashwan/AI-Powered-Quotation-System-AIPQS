import os
from src.object_detection import detect_objects
from src.quotation_generator import QuotationGenerator
from src.report_generator import ReportGenerator

def main(image_path, model_path='models/yolov8n_trained.pt', output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run object detection
    detections_dict = detect_objects(image_path, model_path)

    # Extract detections list for the single image
    detections = []
    if isinstance(image_path, str):
        detections = detections_dict.get(image_path, [])

    # Generate quotation
    qg = QuotationGenerator()
    quotation = qg.generate_quotation(detections)

    # Generate PDF report with quotation number in filename
    quotation_number = quotation.get('download_count', None)
    filename = f"quotation_{quotation_number}.pdf" if quotation_number is not None else "quotation.pdf"
    report_path = os.path.join(output_dir, filename)
    rg = ReportGenerator(filename=report_path)
    rg.generate_pdf(quotation)

    print(f"Quotation report generated at: {report_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full pipeline: detection, quotation, report generation")
    parser.add_argument('image_path', type=str, help='Path to blueprint image')
    parser.add_argument('--model_path', type=str, default='models/yolov8n_trained.pt', help='Path to trained YOLO model')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the PDF report')

    args = parser.parse_args()

    main(args.image_path, args.model_path, args.output_dir)
