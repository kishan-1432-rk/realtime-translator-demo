#!/usr/bin/env python3
"""
Setup script for Demo ASR Model
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def test_installation():
    """Test if the installation was successful"""
    print("Testing installation...")
    try:
        import torch
        import transformers
        import soundfile
        import numpy
        print("✅ All required packages are available!")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        return False

def run_tests():
    """Run the test suite"""
    print("Running tests...")
    try:
        subprocess.check_call([sys.executable, "-m", "pytest", "test_asr_model.py", "-v"])
        print("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Some tests failed: {e}")
        return False

def main():
    """Main setup function"""
    print("=== Demo ASR Model Setup ===")
    
    # Install requirements
    if not install_requirements():
        print("Setup failed during package installation.")
        return
    
    # Test installation
    if not test_installation():
        print("Setup failed during installation test.")
        return
    
    # Run tests
    if not run_tests():
        print("Setup completed with test failures.")
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\nTo run the demo:")
    print("  python3 Demo_ASR_model.py")
    print("\nTo run tests:")
    print("  python3 -m pytest test_asr_model.py -v")

if __name__ == "__main__":
    main()
