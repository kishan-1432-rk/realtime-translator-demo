# Demo ASR Model - Indian Languages

This project implements Automatic Speech Recognition (ASR) for Indian languages using AI4Bharat models. It supports Hindi, Tamil, and Gujarati languages.

## Features

- **Multi-language Support**: Hindi (hi), Tamil (ta), Gujarati (gu)
- **Easy-to-use API**: Simple class-based interface
- **Comprehensive Testing**: Full test suite with pytest
- **Error Handling**: Robust error handling for various scenarios
- **Authentication**: Built-in Hugging Face authentication

## Prerequisites

- Python 3.8 or higher
- Internet connection for downloading models
- Hugging Face account (for authentication)

## Installation

### Quick Setup

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the setup script** (optional):
   ```bash
   python3 setup.py
   ```

### Manual Installation

If you prefer manual installation:

```bash
# Install core dependencies
pip install torch transformers datasets soundfile accelerate

# Install additional dependencies
pip install huggingface-hub numpy librosa pytest
```

## Usage

### Basic Usage

```python
from Demo_ASR_model import IndicASR

# Initialize ASR for Hindi
asr = IndicASR(language="hi")

# Transcribe an audio file
transcription = asr.transcribe("path/to/audio.wav")
print(transcription)
```

### Supported Languages

- `"hi"` - Hindi
- `"ta"` - Tamil  
- `"gu"` - Gujarati

### Running the Demo

```bash
python3 Demo_ASR_model.py
```

This will:
1. Authenticate with Hugging Face
2. Create a test audio file
3. Test transcription with all supported languages
4. Clean up temporary files

### API Reference

#### IndicASR Class

**Constructor:**
```python
IndicASR(language="hi")
```

**Methods:**

- `transcribe(audio_path)`: Transcribe speech from an audio file
- `transcribe_audio_data(audio_data, sampling_rate)`: Transcribe speech from audio data

## Testing

### Run All Tests

```bash
python3 -m pytest test_asr_model.py -v
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end functionality
- **Error Handling**: Exception scenarios
- **Mock Tests**: Offline testing capabilities

### Test Coverage

- Model initialization
- Language validation
- Audio file processing
- Error handling
- Device selection
- Authentication

## File Structure

```
asr/
├── Demo_ASR_model.py      # Main ASR implementation
├── test_asr_model.py      # Test suite
├── requirements.txt       # Python dependencies
├── setup.py              # Setup script
├── context.md            # Project context
├── README.md             # This file
└── Deep_ASR_Demo_with_AI4Bharat.ipynb  # Original notebook
```

## Configuration

### Hugging Face Authentication

The project uses a pre-configured Hugging Face token. If you need to use your own:

1. Get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Update the `HF_TOKEN` variable in `Demo_ASR_model.py`

### Model Selection

Models are automatically selected based on language:
- Hindi: `ai4bharat/indic-whisper-v2-hi`
- Tamil: `ai4bharat/indic-whisper-v2-ta`
- Gujarati: `ai4bharat/indic-whisper-v2-gu`

## Troubleshooting

### Common Issues

1. **Model Loading Fails**
   - Check internet connection
   - Verify Hugging Face authentication
   - Ensure sufficient disk space

2. **Audio File Issues**
   - Use WAV format for best compatibility
   - Ensure audio file exists and is readable
   - Check audio file integrity

3. **Memory Issues**
   - Models will use CPU if CUDA is not available
   - Consider using smaller audio files for testing

### Error Messages

- `"Unsupported language"`: Use one of: 'hi', 'ta', 'gu'
- `"Audio file not found"`: Check file path and permissions
- `"Authentication failed"`: Verify Hugging Face token

## Performance

- **CPU Mode**: Slower but works on all systems
- **GPU Mode**: Faster with CUDA-compatible GPU
- **Memory Usage**: ~2-4GB RAM per model
- **Processing Speed**: ~1-5 seconds per minute of audio (GPU)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is based on AI4Bharat models and follows their licensing terms.

## Acknowledgments

- AI4Bharat for the Indic Whisper models
- Hugging Face for the transformers library
- The open-source community for supporting libraries

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review test cases for examples
3. Check Hugging Face model documentation
