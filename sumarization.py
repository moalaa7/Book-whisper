import os
import fitz  # PyMuPDF
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from waitress import serve
import tempfile
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_CHUNK_SIZE = 1024  # Optimal chunk size for BART
MIN_SUMMARY_LENGTH = 30
MAX_SUMMARY_LENGTH = 130

app = Flask(__name__)
CORS(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise

def get_summarizer():
    # Lazy loading of the model
    if not hasattr(get_summarizer, 'model'):
        get_summarizer.model = pipeline("summarization", model="facebook/bart-large-cnn")
    return get_summarizer.model

def summarize_text(text):
    if not text.strip():
        return "No text found in document."
    
    try:
        summarizer = get_summarizer()
        chunks = [text[i:i+MAX_CHUNK_SIZE] for i in range(0, len(text), MAX_CHUNK_SIZE)]
        summary = []
        
        for chunk in chunks:
            if len(chunk.strip()) < 50:  # Skip very small chunks
                continue
            result = summarizer(chunk, 
                              max_length=MAX_SUMMARY_LENGTH, 
                              min_length=MIN_SUMMARY_LENGTH, 
                              do_sample=False)
            summary.append(result[0]['summary_text'])
        
        return " ".join(summary).strip()
    except Exception as e:
        logger.error(f"Error during summarization: {str(e)}")
        raise

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PDF files are allowed'}), 400
    
    # Create a temporary file
    temp_dir = tempfile.mkdtemp()
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(temp_dir, filename)
        
        # Check file size
        file_content = file.read()
        if len(file_content) > MAX_FILE_SIZE:
            return jsonify({'error': 'File too large. Maximum size is 10MB'}), 400
        
        # Save the file
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Process the file
        text = extract_text_from_pdf(file_path)
        summary = summarize_text(text)
        
        return jsonify({'summary': summary})
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Cleanup
        try:
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

if __name__ == '__main__':
    logger.info("Starting summarization service...")
    serve(app, host='0.0.0.0', port=5001)
