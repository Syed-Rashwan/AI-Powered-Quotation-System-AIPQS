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

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'uploads')
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

# -- Allowed file --  
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

            # Run YOLOv8 inference for rooms and windows using local YOLO model
            try:
                detections = run_inference(filepath, model_path='models/yolov8n_trained.pt')
            except Exception as e:
                flash(f"YOLOv8 inference failed: {str(e)}")
                logger.error(f"YOLOv8 inference failed: {str(e)}")
                return redirect(request.url)

            # Run OCR pipeline to detect room names
            try:
                room_names, bounding_boxes, _ = detect_rooms(filepath, show_image=False)
                logger.info(f"OCR detected room names: {room_names}")

                # Normalize room names using ROOM_SYNONYMS before counting devices
                from src.ocr_blueprint_pipeline import ROOM_SYNONYMS

                normalized_room_names = []
                for room in room_names:
                    normalized_room = ROOM_SYNONYMS.get(room, room)
                    normalized_room_names.append(normalized_room)

                logger.info(f"Normalized room names: {normalized_room_names}")

                # Count devices based on normalized room names
                room_counts = {}
                for room in normalized_room_names:
                    room_counts[room] = room_counts.get(room, 0) + 1

                logger.info(f"Room counts before device counting: {room_counts}")

            except Exception as e:
                flash(f"OCR pipeline failed: {str(e)}")
                logger.error(f"OCR pipeline failed: {str(e)}")
                return redirect(request.url)

            # Count devices based on normalized room names
            room_counts = {}
            for room in normalized_room_names:
                room_counts[room] = room_counts.get(room, 0) + 1
            device_counts = count_devices(room_counts)

            # Add devices for windows and doors detected by YOLO
            # Check if detections is defined before using it
            if 'detections' in locals() and isinstance(detections, list):
                # Map YOLO class IDs to class names
                yolo_class_id_to_name = {
                    0: 'door',
                    1: 'window',
                    2: 'smart curtain',
                    3: 'sensor',
                    # Add other class mappings as needed
                }
                window_count = sum(1 for det in detections if yolo_class_id_to_name.get(det.get('class_id')) == 'window')
                door_count = sum(1 for det in detections if yolo_class_id_to_name.get(det.get('class_id')) == 'door')
                smart_curtain_count = sum(1 for det in detections if yolo_class_id_to_name.get(det.get('class_id')) == 'smart curtain')
                sensor_count = sum(1 for det in detections if yolo_class_id_to_name.get(det.get('class_id')) == 'sensor')
            else:
                window_count = 0
                door_count = 0
                smart_curtain_count = 0
                sensor_count = 0
            device_counts['smart curtain'] = smart_curtain_count
            device_counts['sensor'] = sensor_count

            # Use fixed device_name_to_class_id mapping consistent with QuotationGenerator pricing_rules
            device_name_to_class_id = {
                'switch': 0,
                'light': 1,
                'electrical outlet': 2,
                'smart curtain': 3,
                'sensor': 4,
                'thermostat': 5,
                'chimney': 6,
                'exhaust fan': 7,
            }
            class_id_to_name = {v: k for k, v in device_name_to_class_id.items()}

            # Aggregate device counts from ROOM_DEVICE_RULES and detected room counts
            aggregated_device_counts = {}
            for room, count in room_counts.items():
                devices = count_devices({room: count})
                for device_name, qty in devices.items():
                    aggregated_device_counts[device_name.lower()] = aggregated_device_counts.get(device_name.lower(), 0) + qty

            # Add YOLO detected devices
            yolo_devices = ['smart curtain', 'sensor']
            for device in yolo_devices:
                aggregated_device_counts[device] = aggregated_device_counts.get(device, 0) + device_counts.get(device, 0)

            # Debug logs to verify device counts and mappings
            logger.info(f"Aggregated device counts: {aggregated_device_counts}")
            logger.info(f"Device name to class ID mapping: {device_name_to_class_id}")

            # Map aggregated device counts to class_ids
            device_counts_mapped = {}
            for device_name, count in aggregated_device_counts.items():
                class_id = device_name_to_class_id.get(device_name)
                if class_id is not None:
                    device_counts_mapped[class_id] = device_counts_mapped.get(class_id, 0) + count
            # Get device counts from OCR-based room detection
            ocr_device_counts = count_devices(room_counts)

            # Get device counts from YOLO detections
            yolo_devices = ['smart curtain', 'sensor']
            yolo_device_counts = {}
            for device in yolo_devices:
                yolo_device_counts[device] = device_counts.get(device, 0)

            # Merge OCR and YOLO device counts by device name
            merged_device_counts = {}
            for device_name, count in ocr_device_counts.items():
                merged_device_counts[device_name] = merged_device_counts.get(device_name, 0) + count
            for device_name, count in yolo_device_counts.items():
                merged_device_counts[device_name] = merged_device_counts.get(device_name, 0) + count

            # Map device names to class IDs for quotation generation
            device_counts_mapped = {}
            for device_name, count in merged_device_counts.items():
                class_id = device_name_to_class_id.get(device_name.lower())
                if class_id is not None:
                    device_counts_mapped[class_id] = device_counts_mapped.get(class_id, 0) + count

            # Debug log device_counts_mapped before storing in session
            logger.info(f"Device counts mapped before storing in session: {device_counts_mapped}")

            qg = QuotationGenerator()
            quotation = qg.generate_quotation_from_device_counts(device_counts_mapped)

            # Prepare data for summary display
            items = []
            unit_prices = quotation.get('unit_prices', {})

            for class_id, quantity in quotation.get('items', {}).items():
                price_per_unit = unit_prices.get(class_id, 0.0)
                device_name = class_id_to_name.get(class_id, str(class_id))
                items.append({
                    'name': device_name,
                    'quantity': quantity,
                    'unit_price': price_per_unit,
                    'total_price': price_per_unit * quantity
                })

            # Calculate subtotal, tax, discount, and total for summary display
            subtotal = sum(item['total_price'] for item in items)
            tax_amount = subtotal * (tax_percent / 100.0)
            discount_amount = subtotal * (discount_percent / 100.0)
            total = subtotal + tax_amount - discount_amount

            # Combine YOLO detections and EasyOCR room detections into a single detections list for quotation
            combined_detections = []

            # Add YOLO detections if available
            if detections:
                combined_detections.extend(detections)

            # Add room detections as dummy detections with class_id for quotation
            for room in room_names:
                # This is a placeholder, adjust as needed
                combined_detections.append({'class_id': None, 'confidence': 1.0, 'bbox': None, 'room_name': room})

            # Cache device counts and other info in session
            session['room_names'] = room_names
            # Convert keys to strings for JSON serialization
            session['device_counts'] = {str(k): v for k, v in device_counts_mapped.items()}
            session['combined_detections'] = combined_detections
            session['filename'] = filename
            session['tax_percent'] = tax_percent
            session['discount_percent'] = discount_percent

            # Render preview without saving PDF
            return render_template('result.html', filename=None, items=items, subtotal=subtotal, tax_amount=tax_amount,
                                   discount_amount=discount_amount, total=total, total_cost=total, quotation=quotation,
                                   class_names=None, blueprint_filename=filename, tax_percent=tax_percent,
                                   discount_percent=discount_percent)

    return render_template('upload.html')

# -- Chatbot --
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
            "Authorization": f"Bearer {AIzaSyB-GIIKVZFckpgr77mF88MffwK0JzIgldM}",
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

# -- Finalize Quotation --
@app.route('/finalize', methods=['POST'])
def finalize_quotation():
    # Retrieve cached combined detections and data from session
    combined_detections = session.get('combined_detections')
    filename = session.get('filename')
    tax_percent = session.get('tax_percent', 10.0)
    discount_percent = session.get('discount_percent', 0.0)
    device_counts_mapped = session.get('device_counts')

    if combined_detections is None or filename is None or device_counts_mapped is None:
        flash("Session expired or missing data. Please upload the blueprint again.")
        logger.warning(f"Session expired or missing data on finalize: combined_detections={combined_detections}, filename={filename}, device_counts={device_counts_mapped}")
        return redirect(url_for('upload_file'))

    # Convert keys back to integers for processing
    device_counts_mapped_int = {int(k): v for k, v in device_counts_mapped.items()}

    # Generate quotation from combined detections
    qg = QuotationGenerator()
    
    # Debug log the device counts
    logger.info(f"Device counts from session: {device_counts_mapped_int}")
    
    quotation = qg.generate_quotation_from_device_counts(device_counts_mapped_int)
    
    # Debug log the quotation data
    logger.info(f"Generated quotation data: {quotation}")

    # Class names mapping for PDF generation
    class_names = {
        0: "Switch",
        1: "Light",
        2: "Electrical Outlet",
        3: "Smart Curtain",
        4: "Sensor",
        5: "Thermostat",
        6: "Chimney",
        7: "Exhaust Fan"
    }

    # Debug log the class names mapping
    logger.info(f"Class names mapping for PDF: {class_names}")

    # Generate PDF report with unique filename
    quotation_number = quotation.get('download_count', None)
    filename_pdf = f"quotation_{quotation_number}.pdf" if quotation_number else "quotation.pdf"
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename_pdf)

    # Provide tax and discount rates to ReportGenerator
    tax_rate = tax_percent / 100.0
    discount_rate = discount_percent / 100.0

    rg = ReportGenerator(filename=pdf_path, tax_rate=tax_rate, discount_rate=discount_rate)
    rg.generate_pdf(quotation, class_names=class_names)

    flash("Quotation finalized and PDF generated. You can now download it.")
    return redirect(url_for('download_report', filename=filename_pdf))

#--Download Report--
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
