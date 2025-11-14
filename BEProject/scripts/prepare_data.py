# scripts/prepare_data.py
import os
import sys
import argparse
import shutil
from pathlib import Path
import librosa
import soundfile as sf
from tqdm import tqdm

def prepare_data(input_dir, output_dir, dysarthric_label='0', clear_label='1'):
    """
    Prepare data by organizing audio files into dysarthric (0) and clear (1) folders
    Args:
        input_dir: Directory containing raw audio files
        output_dir: Output directory
        dysarthric_label: Label/folder name for dysarthric speech
        clear_label: Label/folder name for clear speech
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    dysarthric_dir = output_path / dysarthric_label
    clear_dir = output_path / clear_label
    dysarthric_dir.mkdir(parents=True, exist_ok=True)
    clear_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if input has subdirectories matching labels
    if (input_path / dysarthric_label).exists() and (input_path / clear_label).exists():
        print("Found labeled subdirectories, organizing by labels...")
        
        # Process dysarthric files
        print("Processing dysarthric files...")
        process_audio_files(
            input_path / dysarthric_label,
            dysarthric_dir,
            target_sr=16000
        )
        
        # Process clear files
        print("Processing clear files...")
        process_audio_files(
            input_path / clear_label,
            clear_dir,
            target_sr=16000
        )
    else:
        print("No labeled subdirectories found.")
        print("Please organize your data as:")
        print(f"  {input_dir}/{dysarthric_label}/ - dysarthric speech files")
        print(f"  {input_dir}/{clear_label}/ - clear speech files")
        return
    
    # Print statistics
    dysarthric_count = len(list(dysarthric_dir.glob("*.wav")))
    clear_count = len(list(clear_dir.glob("*.wav")))
    
    print("\n" + "=" * 60)
    print("Data Preparation Complete!")
    print(f"Dysarthric files: {dysarthric_count}")
    print(f"Clear files: {clear_count}")
    print(f"Output directory: {output_path}")
    print("=" * 60)

def process_audio_files(input_dir, output_dir, target_sr=16000):
    """Process audio files: resample, convert to mono, normalize"""
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(input_dir.glob(f"*{ext}"))
    
    for audio_file in tqdm(list(audio_files), desc=f"Processing {input_dir.name}"):
        try:
            # Load audio
            audio, sr = librosa.load(audio_file, sr=target_sr, mono=True)
            
            # Skip very short files
            if len(audio) < target_sr * 0.5:  # Less than 0.5 seconds
                print(f"Skipping {audio_file.name} (too short)")
                continue
            
            # Normalize
            audio = librosa.util.normalize(audio)
            
            # Save as WAV
            output_file = output_dir / f"{audio_file.stem}.wav"
            sf.write(output_file, audio, target_sr)
            
        except Exception as e:
            print(f"Error processing {audio_file.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Prepare audio data for training')
    parser.add_argument('--input', required=True, help='Input directory with raw audio')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--dysarthric-label', default='0', help='Label for dysarthric speech')
    parser.add_argument('--clear-label', default='1', help='Label for clear speech')
    args = parser.parse_args()
    
    prepare_data(
        args.input,
        args.output,
        args.dysarthric_label,
        args.clear_label
    )

if __name__ == "__main__":
    main()
