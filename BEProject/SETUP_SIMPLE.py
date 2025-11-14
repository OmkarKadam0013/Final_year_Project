# SETUP_SIMPLE.py
"""
Simple setup script - Just run this!
Place this file in your project root folder.
"""
import os
import sys
import subprocess

def run_command(cmd, description):
    """Run a command and show output"""
    print(f"\n{'='*60}")
    print(f"⏳ {description}...")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True, 
                              capture_output=True)
        print(result.stdout)
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  Dysarthric Speech Conversion - Simple Setup            ║
    ║  This will set up everything automatically              ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Step 1: Install backend dependencies
    if not run_command(
        "pip install torch torchaudio numpy librosa soundfile scipy fastapi uvicorn websockets transformers tensorboard tqdm pyyaml python-dotenv",
        "Installing Python packages"
    ):
        print("⚠️  You may need to run: pip install --upgrade pip")
        return
    
    # Step 2: Create directories
    print("\n📁 Creating directories...")
    os.makedirs("data/raw/0", exist_ok=True)
    os.makedirs("data/raw/1", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    print("✅ Directories created")
    
    # Step 3: Create __init__.py files
    print("\n📝 Creating __init__.py files...")
    init_dirs = [
        "backend/app",
        "backend/app/models",
        "backend/app/preprocessing",
        "backend/app/training",
        "backend/app/utils"
    ]
    for dir_path in init_dirs:
        init_file = os.path.join(dir_path, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("")
            print(f"  Created: {init_file}")
    print("✅ __init__.py files created")
    
    # Step 4: Check if data exists
    print("\n📊 Checking for data...")
    dysarthric_files = len([f for f in os.listdir("data/raw/0") if f.endswith('.wav')])
    clear_files = len([f for f in os.listdir("data/raw/1") if f.endswith('.wav')])
    
    print(f"  Dysarthric files: {dysarthric_files}")
    print(f"  Clear files: {clear_files}")
    
    if dysarthric_files == 0 or clear_files == 0:
        print("\n⚠️  No audio files found!")
        print("   You need to add .wav files to:")
        print("   - data/raw/0/  (dysarthric speech)")
        print("   - data/raw/1/  (clear speech)")
        print("\n   Or generate test data by running:")
        print("   python scripts/generate_test_data.py")
        return
    
    # Success!
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  ✅ SETUP COMPLETE!                                      ║
    ╚══════════════════════════════════════════════════════════╝
    
    Your system is ready!
    
    📋 NEXT STEPS:
    
    1️⃣  Train the model on your data:
       python scripts/train.py --epochs 10 --batch-size 4
       
       This will:
       - Read audio from data/raw/0 and data/raw/1
       - Train the neural network
       - Save models to checkpoints/
       - Takes: ~2 hours for 10 epochs (GPU) or ~10 hours (CPU)
    
    2️⃣  Test conversion on one file:
       python scripts/inference.py --input data/raw/0/sample.wav --output outputs/ --checkpoint checkpoints/best_model.pt
    
    3️⃣  Start the web server:
       python -m backend.app.main
       
       Then open: http://localhost:8000
    
    📖 Need help? Check README.md
    """)

if __name__ == "__main__":
    main()
