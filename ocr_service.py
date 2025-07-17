from flask import Flask, request, send_file, render_template, flash, jsonify
import pytesseract
from PIL import Image
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os
from werkzeug.utils import secure_filename
import logging
import sys
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flash messages

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Try to find Tesseract installation
def find_tesseract():
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Tesseract-OCR\tesseract.exe',
    ]
    
    # Check if tesseract is in PATH
    try:
        pytesseract.get_tesseract_version()
        logger.info("Tesseract found in PATH")
        return True
    except Exception:
        pass
    
    # Check possible installation paths
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            logger.info(f"Tesseract found at: {path}")
            return True
    
    logger.error("Tesseract not found in PATH or common installation locations")
    return False

# Try to configure Tesseract
if not find_tesseract():
    logger.error("""
    Tesseract is not installed or not found. Please:
    1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
    2. Install it to the default location (C:\\Program Files\\Tesseract-OCR)
    3. Make sure to check 'Add to system PATH' during installation
    4. Restart your computer after installation
    """)
    sys.exit(1)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_image(image_path):
    try:
        logger.info(f"Attempting to open image: {image_path}")
        # Open the image using PIL
        img = Image.open(image_path)
        
        # Log image details
        logger.info(f"Image opened successfully. Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
        
        # Convert image to RGB if it's not
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Extract text using pytesseract
        logger.info("Starting OCR process...")
        text = pytesseract.image_to_string(img)
        
        if not text.strip():
            logger.warning("No text was extracted from the image")
            return "No text could be extracted from the image. Please ensure the image contains clear, readable text."
        
        logger.info(f"Successfully extracted text. Length: {len(text)} characters")
        return text
    except Exception as e:
        error_msg = f"Error during text extraction: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return f"Error: {error_msg}"

def create_pdf(text, filename):
    try:
        logger.info("Creating PDF from extracted text")
        # Create a PDF in memory
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Add text to PDF
        y = height - 50  # Start from top of page
        for line in text.split('\n'):
            if y < 50:  # If we're near the bottom of the page
                c.showPage()  # Start a new page
                y = height - 50  # Reset y to top of new page
            c.drawString(50, y, line)
            y -= 15  # Move down for next line
        
        c.save()
        buffer.seek(0)
        logger.info("PDF created successfully")
        return buffer
    except Exception as e:
        error_msg = f"Error creating PDF: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise Exception(error_msg)

@app.route('/ocr', methods=['GET', 'POST'])
def ocr_page():
    if request.method == 'POST':
        logger.info("Received POST request for OCR")
        
        try:
            if 'file' not in request.files:
                logger.error("No file part in request")
                return 'No file uploaded', 400
            
            file = request.files['file']
            if file.filename == '':
                logger.error("No file selected")
                return 'No file selected', 400
            
            if not allowed_file(file.filename):
                logger.error(f"Invalid file type: {file.filename}")
                return 'Invalid file type. Please upload an image file (PNG, JPG, JPEG, or GIF)', 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Ensure upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            logger.info(f"Saving uploaded file to: {filepath}")
            file.save(filepath)
            
            # Extract text from image
            extracted_text = extract_text_from_image(filepath)
            
            if extracted_text.startswith("Error:"):
                # Clean up the uploaded file
                os.remove(filepath)
                return extracted_text, 400
            
            # Create PDF
            try:
                pdf_buffer = create_pdf(extracted_text, filename)
            except Exception as e:
                logger.error(f"Error creating PDF: {str(e)}")
                os.remove(filepath)
                return str(e), 500
            
            # Clean up the uploaded file
            os.remove(filepath)
            
            logger.info("Sending PDF file to user")
            return send_file(
                pdf_buffer,
                as_attachment=True,
                download_name='extracted_text.pdf',
                mimetype='application/pdf'
            )
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            logger.error(traceback.format_exc())
            return f"An unexpected error occurred: {str(e)}", 500
    
    logger.info("Rendering OCR page")
    return render_template('ocr.html')

@app.route('/')
def home():
    return render_template('ocr.html')

@app.errorhandler(413)
def request_entity_too_large(error):
    return 'File too large. Maximum file size is 16MB.', 413

if __name__ == '__main__':
    # Create uploads folder if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.info("Starting Flask application")
    app.run(debug=True) 