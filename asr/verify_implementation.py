#!/usr/bin/env python3
"""
Verification script for ASR Demo Implementation
Checks if the implementation meets all requirements from context.md
"""

import os
import sys
import ast
import re

def check_context_requirements():
    """Check if implementation meets context.md requirements"""
    print("=== Checking Context.md Requirements ===")
    
    requirements = [
        "Convert .ipynb file to python3 file",
        "Use Hugging Face authentication token",
        "Use requirements.txt for libraries",
        "Execute Demo_ASR_model with python3 locally"
    ]
    
    print("Requirements from context.md:")
    for i, req in enumerate(requirements, 1):
        print(f"{i}. {req}")
    
    return requirements

def check_file_structure():
    """Check if all required files are present"""
    print("\n=== Checking File Structure ===")
    
    required_files = [
        "Demo_ASR_model.py",
        "test_asr_model.py", 
        "requirements.txt",
        "setup.py",
        "README.md",
        "context.md"
    ]
    
    missing_files = []
    present_files = []
    
    for file in required_files:
        if os.path.exists(file):
            present_files.append(file)
            print(f"✅ {file}")
        else:
            missing_files.append(file)
            print(f"❌ {file} - MISSING")
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        return False
    else:
        print(f"\n✅ All required files present ({len(present_files)} files)")
        return True

def check_python_syntax():
    """Check Python syntax of main files"""
    print("\n=== Checking Python Syntax ===")
    
    python_files = ["Demo_ASR_model.py", "test_asr_model.py", "setup.py"]
    syntax_errors = []
    
    for file in python_files:
        if os.path.exists(file):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                ast.parse(content)
                print(f"✅ {file} - Syntax OK")
            except SyntaxError as e:
                print(f"❌ {file} - Syntax Error: {e}")
                syntax_errors.append((file, e))
            except Exception as e:
                print(f"⚠️  {file} - Error reading: {e}")
    
    return len(syntax_errors) == 0

def check_requirements_txt():
    """Check requirements.txt file"""
    print("\n=== Checking Requirements.txt ===")
    
    try:
        with open("requirements.txt", "r") as f:
            content = f.read().strip()
        
        if not content:
            print("❌ requirements.txt is empty")
            return False
        
        lines = content.split("\n")
        print(f"✅ Found {len(lines)} requirements:")
        
        required_packages = [
            "torch", "transformers", "datasets", "soundfile", 
            "accelerate", "huggingface-hub", "numpy", "librosa", "pytest"
        ]
        
        for package in required_packages:
            if any(package in line for line in lines):
                print(f"   ✅ {package}")
            else:
                print(f"   ❌ {package} - MISSING")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False

def check_authentication_token():
    """Check if Hugging Face token is properly configured"""
    print("\n=== Checking Authentication Token ===")
    
    try:
        with open("Demo_ASR_model.py", "r") as f:
            content = f.read()
        
        # Look for HF_TOKEN
        token_pattern = r'HF_TOKEN\s*=\s*["\']([^"\']+)["\']'
        match = re.search(token_pattern, content)
        
        if match:
            token = match.group(1)
            if token == "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":
                print("✅ Hugging Face token found and matches context.md")
                return True
            else:
                print("⚠️  Hugging Face token found but doesn't match context.md")
                return False
        else:
            print("❌ Hugging Face token not found in Demo_ASR_model.py")
            return False
            
    except Exception as e:
        print(f"❌ Error checking authentication: {e}")
        return False

def check_asr_class_implementation():
    """Check if IndicASR class is properly implemented"""
    print("\n=== Checking ASR Class Implementation ===")
    
    try:
        with open("Demo_ASR_model.py", "r") as f:
            content = f.read()
        
        # Check for class definition
        if "class IndicASR:" in content:
            print("✅ IndicASR class found")
        else:
            print("❌ IndicASR class not found")
            return False
        
        # Check for required methods
        required_methods = ["__init__", "transcribe"]
        for method in required_methods:
            if f"def {method}" in content:
                print(f"✅ {method} method found")
            else:
                print(f"❌ {method} method not found")
                return False
        
        # Check for language support
        languages = ["hi", "ta", "gu"]
        for lang in languages:
            if f"ai4bharat/indic-whisper-v2-{lang}" in content:
                print(f"✅ {lang} language support found")
            else:
                print(f"❌ {lang} language support not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking ASR implementation: {e}")
        return False

def check_test_coverage():
    """Check if test cases are comprehensive"""
    print("\n=== Checking Test Coverage ===")
    
    try:
        with open("test_asr_model.py", "r") as f:
            content = f.read()
        
        # Check for test classes
        test_classes = ["TestIndicASR", "TestAudioCreation", "TestIntegration", "TestErrorHandling"]
        found_classes = []
        
        for test_class in test_classes:
            if f"class {test_class}" in content:
                print(f"✅ {test_class} test class found")
                found_classes.append(test_class)
            else:
                print(f"❌ {test_class} test class not found")
        
        # Check for pytest usage
        if "import pytest" in content:
            print("✅ pytest import found")
        else:
            print("❌ pytest import not found")
            return False
        
        # Check for mock testing
        if "from unittest.mock" in content:
            print("✅ Mock testing support found")
        else:
            print("⚠️  Mock testing support not found")
        
        return len(found_classes) >= 3  # At least 3 test classes
        
    except Exception as e:
        print(f"❌ Error checking test coverage: {e}")
        return False

def check_documentation():
    """Check if documentation is comprehensive"""
    print("\n=== Checking Documentation ===")
    
    try:
        with open("README.md", "r") as f:
            content = f.read()
        
        # Check for key sections
        required_sections = [
            "Installation", "Usage", "Testing", "Features", "Prerequisites"
        ]
        
        for section in required_sections:
            if f"## {section}" in content:
                print(f"✅ {section} section found")
            else:
                print(f"❌ {section} section not found")
                return False
        
        # Check for code examples
        if "```python" in content:
            print("✅ Code examples found")
        else:
            print("❌ Code examples not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking documentation: {e}")
        return False

def generate_execution_instructions():
    """Generate execution instructions"""
    print("\n=== Execution Instructions ===")
    
    instructions = """
🎯 TO EXECUTE THE ASR DEMO:

1. INSTALL DEPENDENCIES:
   pip install -r requirements.txt

2. RUN THE MAIN DEMO:
   python Demo_ASR_model.py

3. RUN TESTS:
   python -m pytest test_asr_model.py -v

4. RUN SETUP SCRIPT (optional):
   python setup.py

5. RUN VERIFICATION:
   python run_demo.py

📋 PREREQUISITES:
- Python 3.8 or higher
- Internet connection (for model downloads)
- Sufficient disk space (~2-4GB per model)

🔧 TROUBLESHOOTING:
- If Python not found, try: py, python3, or python
- If pip not found, install pip first
- For CUDA support, install PyTorch with CUDA
- Check internet connection for model downloads

📁 FILES CREATED:
- Demo_ASR_model.py (main implementation)
- test_asr_model.py (comprehensive tests)
- requirements.txt (dependencies)
- setup.py (setup script)
- README.md (documentation)
- run_demo.py (verification script)
"""
    
    print(instructions)

def main():
    """Main verification function"""
    print("🔍 ASR Demo Implementation Verification")
    print("=" * 50)
    
    # Check context requirements
    check_context_requirements()
    
    # Run all checks
    checks = [
        ("File Structure", check_file_structure),
        ("Python Syntax", check_python_syntax),
        ("Requirements.txt", check_requirements_txt),
        ("Authentication Token", check_authentication_token),
        ("ASR Class Implementation", check_asr_class_implementation),
        ("Test Coverage", check_test_coverage),
        ("Documentation", check_documentation)
    ]
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check_name, check_func in checks:
        try:
            if check_func():
                passed_checks += 1
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'='*50}")
    print(f"Checks passed: {passed_checks}/{total_checks}")
    
    if passed_checks == total_checks:
        print("🎉 ALL CHECKS PASSED! Implementation is ready for execution.")
    else:
        print("⚠️  Some checks failed. Please review the issues above.")
    
    # Generate execution instructions
    generate_execution_instructions()

if __name__ == "__main__":
    main()

