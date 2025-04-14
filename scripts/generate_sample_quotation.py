import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from quotation_generator import QuotationGenerator
from report_generator import ReportGenerator

def generate_sample_quotation():
    # Mock detections: list of dicts with 'class_id', 'confidence', 'bbox' keys
    mock_detections = [
        {'class_id': 0, 'confidence': 0.9, 'bbox': [0,0,0,0]},
        {'class_id': 1, 'confidence': 0.85, 'bbox': [0,0,0,0]},
        {'class_id': 0, 'confidence': 0.95, 'bbox': [0,0,0,0]},
        {'class_id': 2, 'confidence': 0.8, 'bbox': [0,0,0,0]},
        {'class_id': 2, 'confidence': 0.75, 'bbox': [0,0,0,0]},
        {'class_id': 2, 'confidence': 0.7, 'bbox': [0,0,0,0]},
    ]

    # Create quotation data
    qg = QuotationGenerator()
    quotation = qg.generate_quotation(mock_detections)

    # Company and client info
    company_info = {
        "Company Name": "ABC Construction Ltd.",
        "Address": "123 Builder St.",
        "City": "Buildtown",
        "Phone": "555-1234",
        "Email": "contact@abcconstruction.com"
    }

    client_info = {
        "Client Name": "XYZ Real Estate",
        "Address": "789 Property Ave.",
        "City": "Estatetown",
        "Phone": "555-5678",
        "Email": "info@xyzrealestate.com"
    }

    # Generate PDF report
    rg = ReportGenerator(filename='sample_quotation.pdf')
    rg.generate_pdf(quotation, company_info=company_info, client_info=client_info)

    print("Sample quotation PDF generated: sample_quotation.pdf")

if __name__ == "__main__":
    generate_sample_quotation()
