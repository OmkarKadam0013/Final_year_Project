# scripts/generate_test_data.py
"""Generate fake audio for testing"""
import numpy as np
import soundfile as sf
from pathlib import Path

def generate_audio(duration=3.0, sr=16000):
    """Generate fake speech-like audio"""
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 150 * t)  # Simple sine wave
    audio += 0.1 * np.random.randn(len(audio))  # Add noise
    return audio / np.max(np.abs(audio))  # Normalize

def main():
    print("Generating test data...")
    
    # Create 10 dysarthric files
    Path("data/raw/0").mkdir(parents=True, exist_ok=True)
    for i in range(10):
        audio = generate_audio()
        sf.write(f"data/raw/0/dysarthric_{i:03d}.wav", audio, 16000)
    
    # Create 10 clear files
    Path("data/raw/1").mkdir(parents=True, exist_ok=True)
    for i in range(10):
        audio = generate_audio()
        sf.write(f"data/raw/1/clear_{i:03d}.wav", audio, 16000)
    
    print("✅ Generated 10 dysarthric + 10 clear audio files")
    print(f"   Dysarthric: data/raw/0/")
    print(f"   Clear: data/raw/1/")

if __name__ == "__main__":
    main()
