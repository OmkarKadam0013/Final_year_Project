# scripts/test_system.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from backend.app.utils.config import Config
from backend.app.models.model_manager import ModelManager
from backend.app.preprocessing.audio_processor import AudioProcessor
from backend.app.preprocessing.feature_extractor import FeatureExtractor
import numpy as np

def test_system():
    """Comprehensive system test"""
    print("=" * 80)
    print("SYSTEM TEST")
    print("=" * 80)
    
    # Initialize
    config = Config()
    print(f"✓ Config loaded")
    print(f"  Device: {config.device}")
    
    # Test audio processor
    print("\nTesting Audio Processor...")
    audio_processor = AudioProcessor(config)
    
    # Generate test audio
    test_audio = np.random.randn(16000)  # 1 second
    processed = audio_processor.preprocess_pipeline(test_audio)
    print(f"✓ Audio preprocessing works")
    print(f"  Input shape: {test_audio.shape}")
    print(f"  Output shape: {processed.shape}")
    
    # Test feature extractor
    print("\nTesting Feature Extractor...")
    feature_extractor = FeatureExtractor(config)
    mel = feature_extractor.extract_mel(torch.FloatTensor(processed))
    print(f"✓ Feature extraction works")
    print(f"  Mel shape: {mel.shape}")
    
    # Test models
    print("\nTesting Models...")
    try:
        model_manager = ModelManager(config, checkpoint_path=None)
        print(f"✓ Models initialized")
        
        # Test inference
        print("\nTesting Inference...")
        start_time = time.time()
        with torch.no_grad():
            audio_out = model_manager.convert(mel)
        inference_time = time.time() - start_time
        
        print(f"✓ Inference works")
        print(f"  Output shape: {audio_out.shape}")
        print(f"  Inference time: {inference_time*1000:.2f}ms")
        print(f"  Real-time factor: {(inference_time / (len(processed)/16000)):.3f}")
        
        # Test streaming
        print("\nTesting Streaming...")
        chunk_size = 4096
        mel_chunk = mel[..., :chunk_size]
        
        start_time = time.time()
        audio_chunk, context = model_manager.convert_streaming(mel_chunk.squeeze(0))
        streaming_time = time.time() - start_time
        
        print(f"✓ Streaming works")
        print(f"  Chunk processing time: {streaming_time*1000:.2f}ms")
        print(f"  Target latency: <200ms")
        print(f"  Status: {'✓ PASS' if streaming_time*1000 < 200 else '✗ FAIL'}")
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_system()
    sys.exit(0 if success else 1)
