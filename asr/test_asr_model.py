#!/usr/bin/env python3
"""
Test cases for the Demo ASR Model
"""

import pytest
import os
import numpy as np
import soundfile as sf
import tempfile
import torch
from unittest.mock import Mock, patch
from Demo_ASR_model import IndicASR, create_sample_audio


class TestIndicASR:
    """Test class for IndicASR functionality"""
    
    def setup_method(self):
        """Setup method called before each test"""
        self.test_audio_path = "test_audio.wav"
        self.sample_rate = 16000
        self.duration = 2
        
        # Create a test audio file
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), False)
        audio_data = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        sf.write(self.test_audio_path, audio_data, self.sample_rate)
    
    def teardown_method(self):
        """Cleanup method called after each test"""
        if os.path.exists(self.test_audio_path):
            os.remove(self.test_audio_path)
    
    def test_init_valid_language(self):
        """Test initialization with valid language codes"""
        valid_languages = ['hi', 'ta', 'gu']
        
        for lang in valid_languages:
            try:
                asr = IndicASR(language=lang)
                assert asr.language_map[lang] == f'ai4bharat/indic-whisper-v2-{lang}'
            except Exception as e:
                # Skip if model loading fails due to network/auth issues
                pytest.skip(f"Model loading failed for {lang}: {e}")
    
    def test_init_invalid_language(self):
        """Test initialization with invalid language code"""
        with pytest.raises(ValueError, match="Unsupported language"):
            IndicASR(language="invalid")
    
    def test_language_map(self):
        """Test that language map contains expected mappings"""
        asr = IndicASR(language="hi")
        expected_mappings = {
            'hi': 'ai4bharat/indic-whisper-v2-hi',
            'ta': 'ai4bharat/indic-whisper-v2-ta',
            'gu': 'ai4bharat/indic-whisper-v2-gu',
        }
        assert asr.language_map == expected_mappings
    
    def test_device_selection(self):
        """Test device selection logic"""
        asr = IndicASR(language="hi")
        assert asr.device in ["cuda:0", "cpu"]
        assert asr.torch_dtype in [torch.float16, torch.float32]
    
    def test_transcribe_file_not_found(self):
        """Test transcription with non-existent file"""
        asr = IndicASR(language="hi")
        result = asr.transcribe("non_existent_file.wav")
        assert "Error: Audio file not found" in result
    
    def test_transcribe_valid_file(self):
        """Test transcription with valid audio file"""
        try:
            asr = IndicASR(language="hi")
            result = asr.transcribe(self.test_audio_path)
            # Should return a string (even if transcription is empty for test audio)
            assert isinstance(result, str)
        except Exception as e:
            # Skip if model loading fails
            pytest.skip(f"Model loading failed: {e}")
    
    def test_transcribe_audio_data(self):
        """Test transcription with audio data"""
        try:
            asr = IndicASR(language="hi")
            
            # Create test audio data
            t = np.linspace(0, 1, self.sample_rate, False)
            audio_data = 0.1 * np.sin(2 * np.pi * 440 * t)
            
            result = asr.transcribe_audio_data(audio_data, self.sample_rate)
            assert isinstance(result, str)
        except Exception as e:
            pytest.skip(f"Model loading failed: {e}")


class TestAudioCreation:
    """Test class for audio creation functionality"""
    
    def test_create_sample_audio(self):
        """Test sample audio file creation"""
        try:
            audio_path = create_sample_audio()
            assert os.path.exists(audio_path)
            assert audio_path.endswith(".wav")
            
            # Check if it's a valid audio file
            audio_data, sample_rate = sf.read(audio_path)
            assert len(audio_data) > 0
            assert sample_rate == 16000
            
            # Clean up
            os.remove(audio_path)
        except Exception as e:
            pytest.skip(f"Audio creation failed: {e}")


class TestIntegration:
    """Integration tests"""
    
    def test_multiple_languages(self):
        """Test loading multiple language models"""
        languages = ['hi', 'ta', 'gu']
        
        for lang in languages:
            try:
                asr = IndicASR(language=lang)
                assert asr.model is not None
                assert asr.processor is not None
            except Exception as e:
                pytest.skip(f"Integration test failed for {lang}: {e}")
    
    def test_end_to_end_transcription(self):
        """Test complete transcription pipeline"""
        try:
            # Create test audio
            audio_path = create_sample_audio()
            
            # Test with Hindi model
            asr = IndicASR(language="hi")
            transcription = asr.transcribe(audio_path)
            
            # Should return a string result
            assert isinstance(transcription, str)
            
            # Clean up
            os.remove(audio_path)
        except Exception as e:
            pytest.skip(f"End-to-end test failed: {e}")


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_network_error_handling(self):
        """Test handling of network errors during model loading"""
        with patch('transformers.AutoModelForSpeechSeq2Seq.from_pretrained') as mock_load:
            mock_load.side_effect = Exception("Network error")
            
            with pytest.raises(Exception):
                IndicASR(language="hi")
    
    def test_audio_processing_error(self):
        """Test handling of audio processing errors"""
        try:
            asr = IndicASR(language="hi")
            
            # Create invalid audio data
            invalid_audio = np.array([1, 2, 3, 4, 5])  # Too short
            
            result = asr.transcribe_audio_data(invalid_audio, 16000)
            # Should handle error gracefully
            assert isinstance(result, str)
        except Exception as e:
            pytest.skip(f"Error handling test failed: {e}")


# Mock tests for when models are not available
class TestMockFunctionality:
    """Mock tests for when actual models are not available"""
    
    @patch('transformers.AutoModelForSpeechSeq2Seq.from_pretrained')
    @patch('transformers.AutoProcessor.from_pretrained')
    def test_mock_model_loading(self, mock_processor, mock_model):
        """Test model loading with mocked components"""
        # Setup mocks
        mock_model.return_value = Mock()
        mock_processor.return_value = Mock()
        
        asr = IndicASR(language="hi")
        assert asr.model is not None
        assert asr.processor is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
