# BookWhisperer 📚

Your Gateway to Smarter Reading - A comprehensive book management and reading enhancement platform.

## 🌟 Features

BookWhisperer offers a suite of powerful features to enhance your reading experience:

### 📖 Personalized Book Recommendations
- Get tailored book suggestions based on your interests and preferences
- Explore new genres and authors through smart recommendations
- Access a curated list of books that match your reading style

### 📝 Smart Summaries
- Get concise summaries of books before diving in
- Save time by previewing content
- Make informed decisions about your next read

### 🔊 Text-to-Speech (TTS)
- Convert text to speech using your own voice
- Listen to books on the go
- Customize voice settings for optimal listening experience

### 📸 Image to PDF & Text Extraction
- Convert images to PDF format
- Extract text from images using OCR technology
- Easily digitize physical documents and books

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd BookWhisperer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

The application consists of multiple services that run on different ports:

1. Main Web Interface: `http://localhost:5000`
2. Recommendation Service: `http://localhost:5003`
3. Text-to-Speech Service: `http://localhost:5002`
4. OCR Service: `http://localhost:5000/ocr`

To start all services, run:
```bash
# Start the main application
python app.py

# Start the recommendation service
cd recommendation
python app.py

# Start the TTS service
cd tts
python app.py
```

## 🛠️ Project Structure

```
BookWhisperer/
├── uploads/           # Directory for uploaded files
├── recommendation/    # Book recommendation service
├── tts/              # Text-to-speech service
├── static/           # Static assets (CSS, JS, images)
├── templates/        # HTML templates
├── cache/           # Cache directory
├── venv/            # Virtual environment
└── README.md        # This file
```

## 💡 Usage

1. **Book Recommendations**
   - Visit `http://localhost:5003`
   - Browse personalized book recommendations
   - Filter and sort recommendations based on preferences

2. **Text-to-Speech**
   - Visit `http://localhost:5002`
   - Upload text or enter content
   - Customize voice settings
   - Generate and download audio

3. **Image to PDF & Text Extraction**
   - Visit `http://localhost:5000/ocr`
   - Upload images
   - Extract text or convert to PDF
   - Download processed files

4. **Book Summaries**
   - Visit `summaries.html`
   - Browse available book summaries
   - Read concise overviews of books

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Thanks to all the open-source libraries and tools that made this project possible
- Special thanks to the Python community for their excellent documentation and support

## 📞 Contact

For any questions or suggestions, please open an issue in the repository.

---

Made with ❤️ for book lovers everywhere 