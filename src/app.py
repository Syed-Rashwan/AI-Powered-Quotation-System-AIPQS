from flask import Flask, request, render_template, send_file, redirect, url_for, flash
import os
from scripts.inference import run_inference
from src.quotation_generator import QuotationGenerator
from src.report_generator import ReportGenerator
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'  # Needed for flashing messages

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'blueprint' not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files['blueprint']
        if file.filename == '':
            flash("No selected file")
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
                return redirect(request.url)

            # Run inference
            detections = run_inference(filepath)

            # Generate quotation
            qg = QuotationGenerator()
            quotation = qg.generate_quotation(detections)

            # Prepare data for summary display
            class_names = {
                0: "switch",
                1: "light",
                2: "electrical outlet"
            }
            items = []
            unit_prices = quotation.get('unit_prices', {})
            for class_id, quantity in quotation.get('items', {}).items():
                items.append({
                    'name': class_names.get(class_id, f"Class {class_id}"),
                    'quantity': quantity,
                    'unit_price': unit_prices.get(class_id, 0.0),
                    'total_price': unit_prices.get(class_id, 0.0) * quantity
                })

            # Calculate subtotal, tax, discount, and total for summary display
            subtotal = sum(unit_prices.get(class_id, 0.0) * quantity for class_id, quantity in quotation.get('items', {}).items())
            tax_amount = subtotal * (tax_percent / 100.0)
            discount_amount = subtotal * (discount_percent / 100.0)
            total = subtotal + tax_amount - discount_amount

            # Render preview without saving PDF
            return render_template('result.html', filename=None, items=items, subtotal=subtotal, tax_amount=tax_amount, 
                                   discount_amount=discount_amount, total=total, total_cost=total, quotation=quotation, 
                                   class_names=class_names, blueprint_filename=filename, tax_percent=tax_percent, 
                                   discount_percent=discount_percent)

    return render_template('upload.html')

from flask import jsonify

@app.route('/finalize', methods=['POST'])
def finalize_quotation():
    data = request.form
    blueprint_filename = data.get('blueprint_filename')
    if not blueprint_filename:
        flash("Blueprint filename missing")
        return redirect(url_for('upload_file'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], blueprint_filename)
    if not os.path.exists(filepath):
        flash("Blueprint file not found")
        return redirect(url_for('upload_file'))

    # Run inference again (or ideally cache the previous result, but for simplicity rerun)
    detections = run_inference(filepath)

    # Generate quotation
    qg = QuotationGenerator()
    quotation = qg.generate_quotation(detections)

    # Prepare data for PDF generation
    class_names = {
        0: "switch",
        1: "light",
        2: "electrical outlet"
    }

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
        return redirect(url_for('upload_file'))

if __name__ == '__main__':
    app.run(debug=True)
