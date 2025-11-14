# backend/app/models/__init__.py
"""Neural network models"""
from .generator import Generator
from .discriminator import MultiScaleDiscriminator
from .ppg_extractor import SimplePPGExtractor
from .speaker_encoder import SpeakerEncoder
from .vocoder import HiFiGANGenerator

__all__ = [
    'Generator',
    'MultiScaleDiscriminator', 
    'SimplePPGExtractor',
    'SpeakerEncoder',
    'HiFiGANGenerator'
]
