import sys
import os

# Add project root to sys.path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import requests
from flask import Flask, request, render_template, send_file, redirect, url_for, flash, session, jsonify
from scripts.inference import run_inference
from src.quotation_generator import QuotationGenerator
from src.report_generator import ReportGenerator
from src.ocr_blueprint_pipeline import detect_rooms, count_devices
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates'))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Use environment variable for secret key or fallback to a default (not recommended for production)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'supersecretkey')

# Gemini API key from environment variable
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'YOUR_GEMINI_API_KEY')
GEMINI_API_URL = "https://gemini.api.endpoint/v1/chat"  # Replace with actual Gemini API endpoint

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'blueprint' not in request.files:
            flash("No file part")
            logger.warning("No file part in request")
            return redirect(request.url)
        file = request.files['blueprint']
        if file.filename == '':
            flash("No selected file")
            logger.warning("No file selected")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Get tax and discount percent from form
            try:
                tax_percent = float(request.form.get('tax_percent', 10.0))
                discount_percent = float(request.form.get('discount_percent', 0.0))
            except ValueError:
                flash("Invalid tax or discount value")
                logger.warning("Invalid tax or discount value")
                return redirect(request.url)

            # Run YOLOv8 inference for rooms and windows using Roboflow model
            try:
                # Use Roboflow model ID "builderformer/10"
                detections = run_inference(filepath, model_id='builderformer/10')
            except Exception as e:
                flash(f"YOLOv8 inference failed: {str(e)}")
                logger.error(f"YOLOv8 inference failed: {str(e)}")
                return redirect(request.url)

            # Run OCR pipeline to detect room names
            try:
                room_names, bounding_boxes, _ = detect_rooms(filepath)
            except Exception as e:
                flash(f"OCR pipeline failed: {str(e)}")
                logger.error(f"OCR pipeline failed: {str(e)}")
                return redirect(request.url)

            # Count devices based on room names
            room_counts = {}
            for room in room_names:
                room_counts[room] = room_counts.get(room, 0) + 1
            device_counts = count_devices(room_counts)

            # Add devices for windows and doors detected by YOLO
            window_count = sum(1 for det in detections if det['class_name'].lower() == 'window')
            door_count = sum(1 for det in detections if det['class_name'].lower() == 'door')
            device_counts['smart curtain'] = window_count
            device_counts['sensor'] = door_count

            # Generate quotation based on device counts
            qg = QuotationGenerator()
            quotation = qg.generate_quotation_from_device_counts(device_counts)

            # Prepare data for summary display
            items = []
            unit_prices = quotation.get('unit_prices', {})
            for device, quantity in device_counts.items():
                price_per_unit = unit_prices.get(device, 0.0)
                items.append({
                    'name': device,
                    'quantity': quantity,
                    'unit_price': price_per_unit,
                    'total_price': price_per_unit * quantity
                })

            # Calculate subtotal, tax, discount, and total for summary display
            subtotal = sum(item['total_price'] for item in items)
            tax_amount = subtotal * (tax_percent / 100.0)
            discount_amount = subtotal * (discount_percent / 100.0)
            total = subtotal + tax_amount - discount_amount

            # Cache detections and device counts in session
            session['detections'] = detections
            session['room_names'] = room_names
            session['device_counts'] = device_counts
            session['filename'] = filename
            session['tax_percent'] = tax_percent
            session['discount_percent'] = discount_percent

            # Render preview without saving PDF
            return render_template('result.html', filename=None, items=items, subtotal=subtotal, tax_amount=tax_amount,
                                   discount_amount=discount_amount, total=total, total_cost=total, quotation=quotation,
                                   class_names=None, blueprint_filename=filename, tax_percent=tax_percent,
                                   discount_percent=discount_percent, detected_features={'detections': detections, 'room_names': room_names})

    return render_template('upload.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    session_id = session.get('session_id')
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id

    try:
        # Use Gemini API for chatbot
        headers = {
            "Authorization": f"Bearer {GEMINI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "session_id": session_id,
            "message": user_message
        }
        import requests
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        assistant_message = data.get("response", "Sorry, I couldn't process your request.")
        return jsonify({'response': assistant_message})
    except Exception as e:
        logger.error(f"Chatbot API error: {str(e)}")
        return jsonify({'error': 'Chatbot API error'}), 500

@app.route('/finalize', methods=['POST'])
def finalize_quotation():
    # Retrieve cached detections and data from session
    device_counts = session.get('device_counts')
    filename = session.get('filename')
    tax_percent = session.get('tax_percent', 10.0)
    discount_percent = session.get('discount_percent', 0.0)

    if not device_counts or not filename:
        flash("Session expired or missing data. Please upload the blueprint again.")
        logger.warning("Session expired or missing data on finalize")
        return redirect(url_for('upload_file'))

    # Generate quotation
    qg = QuotationGenerator()
    quotation = qg.generate_quotation_from_device_counts(device_counts)

    # Prepare data for PDF generation
    class_names = None

    # Generate PDF report with unique filename
    quotation_number = quotation.get('download_count', None)
    filename_pdf = f"quotation_{quotation_number}.pdf" if quotation_number else "quotation.pdf"
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename_pdf)
    rg = ReportGenerator(filename=pdf_path)
    rg.generate_pdf(quotation, class_names=class_names)

    flash("Quotation finalized and PDF generated. You can now download it.")
    return redirect(url_for('download_report', filename=filename_pdf))

@app.route('/download/<filename>')
def download_report(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    else:
        flash("Report not found")
        logger.warning(f"Report not found: {filename}")
        return redirect(url_for('upload_file'))

if __name__ == '__main__':
    app.run(debug=True)
