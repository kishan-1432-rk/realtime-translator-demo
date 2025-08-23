#!/usr/bin/env python3
"""
Demo ASR Model - Automatic Speech Recognition for Indian Languages
Converted from Jupyter notebook to Python script
"""

import os
import sys
import torch
import soundfile as sf
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from huggingface_hub import login
import warnings
warnings.filterwarnings("ignore")

# Hugging Face authentication token
HF_TOKEN = "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"

class IndicASR:
    """
    Automatic Speech Recognition class for Indian languages using AI4Bharat models.
    Supports Hindi, Tamil, and Gujarati languages.
    """
    
    def __init__(self, language="hi"):
        """
        Initialize the ASR model for a specific language.

        Args:
            language (str): Language code ('hi' for Hindi, 'ta' for Tamil, 'gu' for Gujarati)
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.language_map = {
            'hi': 'ai4bharat/indic-whisper-v2-hi',
            'ta': 'ai4bharat/indic-whisper-v2-ta', 
            'gu': 'ai4bharat/indic-whisper-v2-gu',
        }
        
        if language not in self.language_map:
            raise ValueError(f"Unsupported language: {language}. Please choose from {list(self.language_map.keys())}")
        
        self.model_id = self.language_map[language]
        print(f"Loading model for {language} from {self.model_id}...")
        
        # Load the pre-trained model and processor
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            token=HF_TOKEN
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            token=HF_TOKEN
        )
        
        print(f"Model loaded successfully on {self.device}")
    
    def transcribe(self, audio_path):
        """
        Transcribe speech from an audio file.

        Args:
            audio_path (str): Path to the audio file (WAV format recommended)

        Returns:
            str: Transcribed text
        """
        try:
            if not os.path.exists(audio_path):
                return f"Error: Audio file not found at {audio_path}"
            
            # Read the audio file
            audio_data, sampling_rate = sf.read(audio_path, dtype='float32')
            
            # Process audio to get input features
            input_features = self.processor(
                audio_data,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            ).input_features.to(self.device, dtype=self.torch_dtype)
            
            # Generate transcription
            predicted_ids = self.model.generate(
                input_features,
                max_new_tokens=128
            )
            
            # Decode predicted IDs to text
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return transcription
            
        except Exception as e:
            return f"An error occurred during transcription: {e}"
    
    def transcribe_audio_data(self, audio_data, sampling_rate):
        """
        Transcribe speech from audio data (numpy array).

        Args:
            audio_data (np.ndarray): Audio data as numpy array
            sampling_rate (int): Sampling rate of the audio

        Returns:
            str: Transcribed text
        """
        try:
            # Process audio to get input features
            input_features = self.processor(
                audio_data,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            ).input_features.to(self.device, dtype=self.torch_dtype)
            
            # Generate transcription
            predicted_ids = self.model.generate(
                input_features,
                max_new_tokens=128
            )
            
            # Decode predicted IDs to text
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return transcription
            
        except Exception as e:
            return f"An error occurred during transcription: {e}"


def create_sample_audio():
    """
    Create a sample audio file for testing purposes.
    """
    import librosa
    
    # Generate a simple sine wave as test audio
    sample_rate = 16000
    duration = 3  # seconds
    frequency = 440  # Hz (A note)
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = 0.1 * np.sin(2 * np.pi * frequency * t)
    
    # Save as WAV file
    test_audio_path = "test_audio.wav"
    sf.write(test_audio_path, audio_data, sample_rate)
    print(f"Created test audio file: {test_audio_path}")
    
    return test_audio_path


def main():
    """
    Main function to demonstrate ASR functionality.
    """
    print("=== Demo ASR Model - Indian Languages ===")
    print("Authenticating with Hugging Face...")
    
    try:
        # Login to Hugging Face
        login(token=HF_TOKEN)
        print("Authentication successful!")
    except Exception as e:
        print(f"Authentication failed: {e}")
        return
    
    # Create test audio file
    test_audio_path = create_sample_audio()
    
    # Test with different languages
    languages = ['hi', 'ta', 'gu']
    
    for lang in languages:
        try:
            print(f"\n--- Testing {lang.upper()} ASR Model ---")
            asr_model = IndicASR(language=lang)
            
            # Transcribe the test audio
            transcription = asr_model.transcribe(test_audio_path)
            print(f"Transcription ({lang}): {transcription}")
            
        except Exception as e:
            print(f"Error with {lang} model: {e}")
    
    # Clean up test file
    if os.path.exists(test_audio_path):
        os.remove(test_audio_path)
        print(f"\nCleaned up test file: {test_audio_path}")


if __name__ == "__main__":
    main()
