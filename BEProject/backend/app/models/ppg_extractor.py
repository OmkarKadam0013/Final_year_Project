# backend/app/models/ppg_extractor.py
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from typing import Optional

class PPGExtractor(nn.Module):
    """Production PPG extractor using pre-trained Wav2Vec2"""
    
    def __init__(self, config, pretrained_model: str = "facebook/wav2vec2-base-960h"):
        super().__init__()
        self.config = config
        
        # Load pre-trained Wav2Vec2
        self.processor = Wav2Vec2Processor.from_pretrained(pretrained_model)
        self.model = Wav2Vec2ForCTC.from_pretrained(pretrained_model)
        
        # Freeze parameters (pre-trained frozen)
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        
        # Projection layer to convert to PPG dimensions
        hidden_size = self.model.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, config.model.ppg_dim * 2),
            nn.ReLU(),
            nn.Linear(config.model.ppg_dim * 2, config.model.ppg_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract PPG from audio
        Args:
            audio: (B, T) or mel spectrogram (B, n_mels, T)
        Returns:
            ppg: (B, ppg_dim, T')
        """
        with torch.no_grad():
            # If mel, convert to audio (simplified - in production use vocoder)
            if audio.dim() == 3:
                # This is mel spectrogram, use it as features
                features = audio.mean(dim=1)  # Average over mel bins
            else:
                # Raw audio
                features = audio
            
            # Get hidden states from Wav2Vec2
            outputs = self.model.wav2vec2(features, output_hidden_states=True)
            hidden_states = outputs.last_hidden_state  # (B, T', hidden_size)
        
        # Project to PPG space
        ppg = self.projection(hidden_states)  # (B, T', ppg_dim)
        ppg = ppg.transpose(1, 2)  # (B, ppg_dim, T')
        
        return ppg
    
    @torch.no_grad()
    def extract_from_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """Extract PPG directly from mel spectrogram"""
        # Simple approach: use mel features directly
        # In production, you'd use inverse mel -> Wav2Vec2
        
        # Temporal pooling/upsampling to match expected length
        B, C, T = mel.shape
        
        # Use convolutional projection
        ppg_features = self.mel_to_ppg_conv(mel)  # Define this layer in __init__
        
        return ppg_features


# Simplified PPG Extractor (for initial training)
class SimplePPGExtractor(nn.Module):
    """Lightweight PPG extractor for training"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(config.audio.n_mels, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Block 2
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Block 3
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Output
            nn.Conv1d(128, config.model.ppg_dim, kernel_size=1)
        )
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, n_mels, T)
        Returns:
            ppg: (B, ppg_dim, T)
        """
        ppg = self.conv_blocks(mel)
        ppg = self.softmax(ppg)
        return ppg
