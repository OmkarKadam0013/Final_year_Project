# scripts/inference.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from pathlib import Path
from backend.app.utils.config import Config
from backend.app.models.model_manager import ModelManager
from backend.app.preprocessing.audio_processor import AudioProcessor
from backend.app.preprocessing.feature_extractor import FeatureExtractor

def main():
    parser = argparse.ArgumentParser(description='Inference for Dysarthric Speech Conversion')
    parser.add_argument('--input', required=True, help='Input audio file or directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device')
    parser.add_argument('--batch', action='store_true', help='Process directory in batch')
    args = parser.parse_args()
    
    # Initialize config
    config = Config()
    config.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Initialize model manager
    print("Loading models...")
    model_manager = ModelManager(config, args.checkpoint)
    print("Models loaded successfully!")
    print(model_manager.get_model_info())
    
    # Initialize processors
    audio_processor = AudioProcessor(config)
    feature_extractor = FeatureExtractor(config)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get input files
    input_path = Path(args.input)
    if input_path.is_file():
        audio_files = [input_path]
    elif input_path.is_dir():
        audio_files = list(input_path.glob("*.wav"))
        audio_files.extend(input_path.glob("*.mp3"))
    else:
        raise ValueError(f"Invalid input path: {args.input}")
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Process files
    for audio_file in audio_files:
        print(f"\nProcessing: {audio_file.name}")
        
        try:
            # Load and preprocess
            audio = audio_processor.preprocess_pipeline(
                audio_processor.load_audio(str(audio_file))
            )
            
            # Extract mel
            mel = feature_extractor.extract_mel(torch.FloatTensor(audio))
            
            # Convert
            print("Converting...")
            audio_clear = model_manager.convert(mel)
            
            # Post-process
            audio_clear_np = audio_clear.squeeze().cpu().numpy()
            audio_clear_np = audio_processor.apply_deemphasis(audio_clear_np)
            
            # Save
            output_file = output_path / f"{audio_file.stem}_clear.wav"
            audio_processor.save_audio(audio_clear_np, str(output_file))
            
            print(f"Saved: {output_file.name}")
            
        except Exception as e:
            print(f"Error processing {audio_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nInference completed!")

if __name__ == "__main__":
    main()
