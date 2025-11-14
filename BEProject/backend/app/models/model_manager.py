# backend/app/models/model_manager.py
import torch
import torch.nn as nn
import os
from typing import Optional, Dict
import numpy as np

class ModelManager:
    """Manages model loading, optimization, and inference"""
    
    def __init__(self, config, checkpoint_path: Optional[str] = None):
        self.config = config
        self.device = config.device
        
        # Initialize models
        self._init_models()
        
        # Load checkpoint
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        # Optimize models for inference
        self._optimize_models()
    
    def _init_models(self):
        """Initialize models for inference"""
        from .generator import Generator
        from .ppg_extractor import SimplePPGExtractor
        from .speaker_encoder import SpeakerEncoder
        from .vocoder import HiFiGANGenerator
        
        print("Initializing inference models...")
        
        # Generator (only I2C for dysarthric to clear)
        self.generator = Generator(self.config).to(self.device)
        
        # PPG extractor
        self.ppg_extractor = SimplePPGExtractor(self.config).to(self.device)
        
        # Speaker encoder
        self.speaker_encoder = SpeakerEncoder(self.config).to(self.device)
        
        # Vocoder
        self.vocoder = HiFiGANGenerator(self.config).to(self.device)
        
        # Set to eval mode
        self.generator.eval()
        self.ppg_extractor.eval()
        self.speaker_encoder.eval()
        self.vocoder.eval()
    
    def _optimize_models(self):
        """Optimize models for inference"""
        if self.config.use_half_precision:
            print("Converting models to FP16...")
            self.generator = self.generator.half()
            self.ppg_extractor = self.ppg_extractor.half()
            self.speaker_encoder = self.speaker_encoder.half()
            self.vocoder = self.vocoder.half()
        
        # Compile models for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            print("Compiling models with torch.compile...")
            try:
                self.generator = torch.compile(self.generator, mode='reduce-overhead')
                self.vocoder = torch.compile(self.vocoder, mode='reduce-overhead')
            except Exception as e:
                print(f"Could not compile models: {e}")
        
        # Quantization for CPU inference
        if self.config.use_quantization and not torch.cuda.is_available():
            print("Quantizing models for CPU inference...")
            self.generator = torch.quantization.quantize_dynamic(
                self.generator, {nn.Linear, nn.Conv1d}, dtype=torch.qint8
            )
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load state dicts
        self.generator.load_state_dict(checkpoint['G_I2C'], strict=False)
        
        if 'PPG_extractor' in checkpoint:
            self.ppg_extractor.load_state_dict(checkpoint['PPG_extractor'], strict=False)
        
        if 'Speaker_encoder' in checkpoint:
            self.speaker_encoder.load_state_dict(checkpoint['Speaker_encoder'], strict=False)
        
        if 'Vocoder' in checkpoint:
            self.vocoder.load_state_dict(checkpoint['Vocoder'], strict=False)
        
        print("Checkpoint loaded successfully")
    
    @torch.no_grad()
    def convert(self, mel_dysarthric: torch.Tensor) -> torch.Tensor:
        """
        Convert dysarthric mel to clear audio
        Args:
            mel_dysarthric: (B, n_mels, T) or (n_mels, T)
        Returns:
            audio_clear: (B, 1, T_audio) or (1, T_audio)
        """
        # Handle single sample
        single_sample = mel_dysarthric.dim() == 2
        if single_sample:
            mel_dysarthric = mel_dysarthric.unsqueeze(0)
        
        # Move to device and convert dtype
        mel_dysarthric = mel_dysarthric.to(self.device)
        if self.config.use_half_precision:
            mel_dysarthric = mel_dysarthric.half()
        
        # Extract PPG and speaker embedding
        ppg = self.ppg_extractor(mel_dysarthric)
        speaker_emb = self.speaker_encoder(mel_dysarthric)
        
        # Generate clear mel
        mel_clear = self.generator(ppg, speaker_emb)
        
        # Convert to audio
        audio_clear = self.vocoder(mel_clear)
        
        if single_sample:
            audio_clear = audio_clear.squeeze(0)
        
        return audio_clear
    
    @torch.no_grad()
    def convert_streaming(self, mel_chunk: torch.Tensor, 
                         context: Optional[Dict] = None) -> tuple:
        """
        Convert audio in streaming mode with context
        Args:
            mel_chunk: (n_mels, T_chunk)
            context: Dictionary containing previous context
        Returns:
            audio_chunk: (1, T_audio)
            new_context: Updated context
        """
        if context is None:
            context = {
                'speaker_emb': None,
                'prev_ppg': None
            }
        
        mel_chunk = mel_chunk.unsqueeze(0).to(self.device)
        if self.config.use_half_precision:
            mel_chunk = mel_chunk.half()
        
        # Extract or reuse speaker embedding (cached for session)
        if context['speaker_emb'] is None:
            context['speaker_emb'] = self.speaker_encoder(mel_chunk)
        
        # Extract PPG
        ppg = self.ppg_extractor(mel_chunk)
        
        # Generate clear mel
        mel_clear = self.generator(ppg, context['speaker_emb'])
        
        # Convert to audio
        audio_clear = self.vocoder(mel_clear)
        
        # Update context
        context['prev_ppg'] = ppg
        
        return audio_clear.squeeze(0), context
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        def count_params(model):
            return sum(p.numel() for p in model.parameters())
        
        return {
            'generator_params': count_params(self.generator),
            'ppg_extractor_params': count_params(self.ppg_extractor),
            'speaker_encoder_params': count_params(self.speaker_encoder),
            'vocoder_params': count_params(self.vocoder),
            'device': str(self.device),
            'half_precision': self.config.use_half_precision,
            'quantized': self.config.use_quantization
        }
