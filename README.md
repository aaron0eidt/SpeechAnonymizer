# 🎙️ Speech Anonymizer

A web application that records or processes uploaded audio files, transcribes them to text using OpenAI Whisper, and anonymizes the text using local models to remove personally identifiable information (PII).

## ✨ Features

- **🎤 Audio Recording**: Record audio directly in your browser
- **📁 File Upload**: Support for multiple audio formats (WAV, MP3, MP4, M4A, FLAC, OGG, WebM)
- **🔊 Speech-to-Text**: High-quality transcription using OpenAI Whisper
- **👥 Speaker Identification**: Automatic speaker diarization (Speaker 1, Speaker 2, etc.)
- **🔒 Text Anonymization**: Anonymization using local models (spaCy, BERT, Ensemble)
- **🎨 Modern UI**: Clean, responsive web interface
- **⚡ Real-time Processing**: Live audio recording with timer

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio processing)

### Installation

1. **Clone or download this project to your local machine**

2. **Install FFmpeg** (required for audio processing):
   - **macOS**: `brew install ffmpeg`
   - **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`
   - **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html)

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Open your browser** and go to `http://localhost:5000`

## 📖 How to Use

### 1. Choose Input Method
- **Record Audio**: Use your microphone to record speech directly
- **Upload File**: Upload an existing audio file

### 2. Transcription
- The application will automatically transcribe your audio using Whisper
- If multiple speakers are detected, they will be labeled as "Speaker 1:", "Speaker 2:", etc.
- The transcribed text will appear in the transcription box
- You can edit the text if needed

### 3. Anonymization
Choose your anonymizer model:
- **spaCy**: Fast and efficient NER-based anonymization.
- **BERT**: Transformer-based model for higher accuracy.
- **Ensemble**: A combination of spaCy and BERT for the best results.

Click "Anonymize Text" to process the transcription.

## 🔧 Configuration

### Supported Audio Formats
- WAV, MP3, MP4, M4A, FLAC, OGG, WebM
- Maximum file size: 16MB

## 🛠️ Development

### Project Structure
```
speech-anonymizer/
├── app.py                 # Main Flask application
├── bert_anonymizer/       # BERT model for anonymization
├── spacy_anonymizer/      # spaCy model for anonymization
├── ensemble_anonymizer/   # Ensemble model for anonymization
├── templates/
│   └── index.html        # Frontend interface
├── uploads/              # Temporary file storage
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

### API Endpoints

- `GET /` - Main application interface
- `POST /upload` - Upload and transcribe audio file
- `POST /transcribe_recording` - Transcribe recorded audio
- `POST /anonymize` - Anonymize transcribed text
- `GET /health` - Health check endpoint

## 🔒 Privacy & Security

- **Local Processing**: Audio transcription happens locally using Whisper
- **Local Anonymization**: Text anonymization is performed locally using your choice of spaCy, BERT, or an ensemble model. No data is sent to external services.
- **No Data Storage**: Uploaded files are automatically deleted after processing
- **Temporary Files**: All temporary files are cleaned up immediately

## 🐛 Troubleshooting

### Common Issues

1. **"Error accessing microphone"**
   - Grant microphone permissions in your browser
   - Use HTTPS in production (required for microphone access)

2. **"Whisper model not loaded"**
   - Ensure you have enough RAM (at least 4GB free)
   - Try using the `tiny` model instead by changing line 26 in `app.py`

3. **Audio transcription errors**
   - Ensure FFmpeg is properly installed
   - Check that your audio file is in a supported format

### Performance Tips

- **Whisper Model Selection**: Change the model size in `app.py` line 26:
  - `tiny` - Fastest, least accurate
  - `base` - Good balance (default)
  - `small` - Better accuracy
  - `medium` - High accuracy
  - `large` - Best accuracy, slowest


