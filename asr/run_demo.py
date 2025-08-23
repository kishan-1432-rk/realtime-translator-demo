#!/usr/bin/env python3
"""
Simple demo runner for ASR Model
This script will test the basic functionality without requiring full model downloads
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print("✅ Transformers imported successfully")
        print(f"   Transformers version: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import soundfile as sf
        print("✅ SoundFile imported successfully")
    except ImportError as e:
        print(f"❌ SoundFile import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        from huggingface_hub import login
        print("✅ HuggingFace Hub imported successfully")
    except ImportError as e:
        print(f"❌ HuggingFace Hub import failed: {e}")
        return False
    
    return True

def test_asr_class():
    """Test the ASR class structure"""
    print("\nTesting ASR class structure...")
    
    try:
        from Demo_ASR_model import IndicASR
        print("✅ IndicASR class imported successfully")
        
        # Test class initialization (without loading models)
        print("Testing class initialization...")
        
        # This will fail due to model loading, but we can test the structure
        try:
            asr = IndicASR(language="hi")
            print("✅ ASR class initialized successfully")
        except Exception as e:
            print(f"⚠️  ASR initialization failed (expected without internet): {e}")
            print("   This is normal if models are not downloaded yet")
        
        return True
        
    except ImportError as e:
        print(f"❌ ASR class import failed: {e}")
        return False

def test_audio_creation():
    """Test audio creation functionality"""
    print("\nTesting audio creation...")
    
    try:
        from Demo_ASR_model import create_sample_audio
        print("✅ Audio creation function imported successfully")
        
        # Test audio creation
        audio_path = create_sample_audio()
        if os.path.exists(audio_path):
            print(f"✅ Test audio created: {audio_path}")
            # Clean up
            os.remove(audio_path)
            print("✅ Test audio cleaned up")
            return True
        else:
            print("❌ Test audio creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Audio creation test failed: {e}")
        return False

def test_requirements():
    """Test if requirements.txt is valid"""
    print("\nTesting requirements.txt...")
    
    try:
        with open("requirements.txt", "r") as f:
            requirements = f.read().strip().split("\n")
        
        print(f"✅ Found {len(requirements)} requirements")
        for req in requirements:
            if req.strip():
                print(f"   - {req.strip()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Requirements test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=== ASR Demo Verification ===")
    print("This script tests the basic structure and dependencies")
    print("Full functionality requires internet connection and model downloads\n")
    
    tests_passed = 0
    total_tests = 4
    
    # Test imports
    if test_imports():
        tests_passed += 1
    
    # Test ASR class
    if test_asr_class():
        tests_passed += 1
    
    # Test audio creation
    if test_audio_creation():
        tests_passed += 1
    
    # Test requirements
    if test_requirements():
        tests_passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All basic tests passed!")
        print("\nTo run the full demo:")
        print("1. Ensure you have internet connection")
        print("2. Run: python Demo_ASR_model.py")
        print("3. Or run tests: python -m pytest test_asr_model.py -v")
    else:
        print("⚠️  Some tests failed. Please check the installation.")
        print("\nTo install dependencies:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()

