# scripts/verify_setup.py
"""Verify all files and setup are correct"""
import os
from pathlib import Path

def check_file(path, description):
    """Check if file exists"""
    if Path(path).exists():
        print(f"✓ {description}")
        return True
    else:
        print(f"✗ {description} - MISSING: {path}")
        return False

def main():
    print("="*60)
    print("Setup Verification")
    print("="*60)
    
    checks = []
    
    # Backend structure
    print("\n[Backend Structure]")
    checks.append(check_file("backend/app/__init__.py", "Backend init"))
    checks.append(check_file("backend/app/main.py", "FastAPI main"))
    checks.append(check_file("backend/app/models/generator.py", "Generator model"))
    checks.append(check_file("backend/app/models/discriminator.py", "Discriminator"))
    checks.append(check_file("backend/app/models/vocoder.py", "Vocoder"))
    checks.append(check_file("backend/app/training/trainer.py", "Trainer"))
    checks.append(check_file("backend/app/preprocessing/audio_processor.py", "Audio processor"))
    checks.append(check_file("backend/requirements.txt", "Requirements"))
    
    # Frontend structure
    print("\n[Frontend Structure]")
    checks.append(check_file("frontend/package.json", "Package.json"))
    checks.append(check_file("frontend/src/App.js", "App.js"))
    checks.append(check_file("frontend/src/components/AudioRecorder.jsx", "AudioRecorder"))
    checks.append(check_file("frontend/src/services/websocket.js", "WebSocket service"))
    
    # Scripts
    print("\n[Scripts]")
    checks.append(check_file("scripts/train.py", "Training script"))
    checks.append(check_file("scripts/inference.py", "Inference script"))
    checks.append(check_file("scripts/generate_test_data.py", "Test data generator"))
    
    # Config files
    print("\n[Configuration]")
    checks.append(check_file("docker-compose.yml", "Docker compose"))
    checks.append(check_file("backend/Dockerfile", "Backend Dockerfile"))
    checks.append(check_file("frontend/Dockerfile", "Frontend Dockerfile"))
    
    # Directories
    print("\n[Directories]")
    for dir_path in ["data/raw/0", "data/raw/1", "checkpoints", "logs"]:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}")
            checks.append(True)
        else:
            print(f"✗ {dir_path} - Create with: mkdir -p {dir_path}")
            checks.append(False)
    
    # Summary
    print("\n" + "="*60)
    passed = sum(checks)
    total = len(checks)
    print(f"Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("✓✓✓ All checks passed! Ready to proceed.")
        return True
    else:
        print("⚠ Some files missing. Review errors above.")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
