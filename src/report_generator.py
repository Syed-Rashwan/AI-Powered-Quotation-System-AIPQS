from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

class ReportGenerator:
    def __init__(self, filename='quotation.pdf', tax_rate=0.1, discount_rate=0.0, terms_and_conditions=None):
        self.filename = filename
        self.tax_rate = tax_rate
        self.discount_rate = discount_rate
        if terms_and_conditions is None:
            self.terms_and_conditions = (
                "Terms and Conditions:\n"
                "1. Payment due within 30 days.\n"
                "2. Goods remain property of the seller until paid in full.\n"
                "3. Warranty as per manufacturer terms.\n"
                "4. Please contact us for any queries."
            )
        else:
            self.terms_and_conditions = terms_and_conditions

    def generate_pdf(self, quotation_data, class_names=None, company_info=None, client_info=None, 
                     quotation_number=None):
        """
        Generates a professional PDF report for the quotation.

        Args:
            quotation_data (dict): Output from QuotationGenerator.generate_quotation
            class_names (dict): Optional mapping from class_id to human-readable names
            company_info (dict): Information about the company (from)
            client_info (dict): Information about the client (to)
            quotation_number (int or None): Unique quotation number to display
        """
        c = canvas.Canvas(self.filename, pagesize=letter)
        width, height = letter

        # Title with background color
        c.setFillColor(colors.HexColor("#003366"))
        c.rect(0, height - inch, width, inch, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 24)
        c.drawString(1 * inch, height - 0.7 * inch, "Quotation Report")

        # Display quotation number below title if provided
        if quotation_number is not None:
            c.setFont("Helvetica-Bold", 18)
            c.setFillColor(colors.red)
            c.drawString(1 * inch, height - 1.1 * inch, f"Quotation {quotation_number}")
            c.setFillColor(colors.black)
            c.setFont("Helvetica-Bold", 12)
            c.drawString(1 * inch, height - 1.5 * inch, "From:")
        else:
            c.setFillColor(colors.black)
            c.setFont("Helvetica-Bold", 12)
            c.drawString(1 * inch, height - 1.5 * inch, "From:")
        c.setFont("Helvetica", 10)
        y = height - 1.7 * inch
        if company_info:
            for key, value in company_info.items():
                c.drawString(1 * inch, y, f"{key}: {value}")
                y -= 0.18 * inch
        else:
            c.drawString(1 * inch, y, "Your Company Name")
            y -= 0.18 * inch
            c.drawString(1 * inch, y, "Address Line 1")
            y -= 0.18 * inch
            c.drawString(1 * inch, y, "Address Line 2")
            y -= 0.18 * inch
            c.drawString(1 * inch, y, "Phone: XXX-XXX-XXXX")
            y -= 0.18 * inch
            c.drawString(1 * inch, y, "Email: info@company.com")
            y -= 0.18 * inch

        # Client Info (To)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(4.5 * inch, height - 1.5 * inch, "To:")
        c.setFont("Helvetica", 10)
        y_client = height - 1.7 * inch
        if client_info:
            for key, value in client_info.items():
                c.drawString(4.5 * inch, y_client, f"{key}: {value}")
                y_client -= 0.18 * inch
        else:
            c.drawString(4.5 * inch, y_client, "Client Name")
            y_client -= 0.18 * inch
            c.drawString(4.5 * inch, y_client, "Client Address Line 1")
            y_client -= 0.18 * inch
            c.drawString(4.5 * inch, y_client, "Client Address Line 2")
            y_client -= 0.18 * inch
            c.drawString(4.5 * inch, y_client, "Phone: XXX-XXX-XXXX")
            y_client -= 0.18 * inch
            c.drawString(4.5 * inch, y_client, "Email: client@example.com")
            y_client -= 0.18 * inch

        # Table of items
        items = quotation_data.get('items', {})
        total_cost = quotation_data.get('total_cost', 0.0)

        if class_names is None:
            class_names = {}

        # Assume unit_prices are available in quotation_data or fallback to 0.0
        unit_prices = quotation_data.get('unit_prices', {})

        data = [['Item', 'Quantity', 'Unit Price', 'Total Price']]
        subtotal = 0.0
        for class_id, quantity in items.items():
            name = class_names.get(class_id, f"Class {class_id}")
            unit_price = unit_prices.get(class_id, 0.0)
            total_price = unit_price * quantity
            subtotal += total_price
            data.append([name, str(quantity), f"${unit_price:.2f}", f"${total_price:.2f}"])

        # Subtotal, taxes, discount, and total rows
        tax_amount = subtotal * self.tax_rate
        discount_amount = subtotal * self.discount_rate
        total = subtotal + tax_amount - discount_amount

        data.append(['', '', 'Subtotal:', f"${subtotal:.2f}"])
        data.append(['', '', f"Tax ({self.tax_rate*100:.0f}%):", f"${tax_amount:.2f}"])
        data.append(['', '', f"Discount ({self.discount_rate*100:.0f}%):", f"-${discount_amount:.2f}"])
        data.append(['', '', 'Total Cost:', f"${total:.2f}"])

        table = Table(data, colWidths=[3*inch, 1*inch, 1.25*inch, 1.25*inch])
        style = TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#003366")),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN',(1,1),(-1,-1),'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f0f8ff")),
        ])
        table.setStyle(style)

        # Position table
        table.wrapOn(c, width, height)
        table_height = 18 * len(data)  # Approx height
        table.drawOn(c, 1 * inch, y_client - table_height - 0.5*inch)

        # Terms and Conditions at the bottom
        text = c.beginText()
        text.setTextOrigin(1 * inch, 1 * inch)
        text.setFont("Helvetica", 9)
        for line in self.terms_and_conditions.split('\n'):
            text.textLine(line)
        c.drawText(text)

        c.showPage()
        c.save()
