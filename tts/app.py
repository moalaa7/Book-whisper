# 




from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import PyPDF2
from TTS.api import TTS
import torch
import tempfile
import numpy as np
from pydub import AudioSegment
import time

# Get the absolute path of the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
# Set upload folder to be inside the tts directory
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize TTS with a voice cloning model
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=True, gpu=torch.cuda.is_available())

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'wav', 'mp3'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def convert_mp3_to_wav(mp3_path):
    """Convert MP3 file to WAV format"""
    wav_path = mp3_path.rsplit('.', 1)[0] + '.wav'
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")
    return wav_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract text based on file type
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        else:
            text = extract_text_from_txt(file_path)
        
        # Clean up the uploaded file
        os.remove(file_path)
        
        return jsonify({
            'text': text
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/clone-voice', methods=['POST'])
def clone_voice():
    if 'voice' not in request.files:
        return jsonify({'error': 'No voice file provided'}), 400

    voice_file = request.files['voice']
    if voice_file.filename == '':
        return jsonify({'error': 'No voice file selected'}), 400

    # Save the uploaded voice sample
    voice_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(voice_file.filename))
    voice_file.save(voice_path)

    # Convert MP3 to WAV if necessary
    if voice_path.lower().endswith('.mp3'):
        voice_path = convert_mp3_to_wav(voice_path)

    text = request.form.get('text', '')
    if not text:
        os.remove(voice_path)  # Clean up voice file
        return jsonify({'error': 'No text provided'}), 400

    try:
        # Generate a unique filename for the output
        output_filename = f'output_{int(time.time())}.wav'
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        # Generate the speech
        tts.tts_to_file(text=text, file_path=output_path, speaker_wav=voice_path, language="en")

        # Clean up the voice file
        os.remove(voice_path)

        # Send the file and then clean it up
        response = send_file(output_path, mimetype='audio/wav')
        
        # Schedule cleanup after sending
        @response.call_on_close
        def cleanup():
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except Exception as e:
                print(f"Error cleaning up file: {e}")
        
        return response

    except Exception as e:
        # Clean up files in case of error
        if os.path.exists(voice_path):
            os.remove(voice_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 