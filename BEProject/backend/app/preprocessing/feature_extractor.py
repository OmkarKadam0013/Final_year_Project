# backend/app/preprocessing/feature_extractor.py
import torch
import torchaudio
import librosa
import numpy as np
from typing import Tuple, Optional


class FeatureExtractor:
    """
    🚀 Production-ready feature extractor (optimized for speed and stability).
    - Uses GPU-accelerated torchaudio ops when available.
    - Ensures minimal CPU↔GPU transfer.
    - Handles fallbacks for NaN / invalid tensors.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Audio / Feature Parameters ----
        self.sample_rate = config.audio.sample_rate
        self.n_fft = config.audio.n_fft
        self.hop_length = config.audio.hop_length
        self.win_length = config.audio.win_length
        self.n_mels = config.audio.n_mels
        self.n_mfcc = config.audio.n_mfcc
        self.fmin = getattr(config.audio, "fmin", 0)
        self.fmax = getattr(config.audio, "fmax", self.sample_rate // 2)

        # ---- Cached transforms (keep on GPU) ----
        self._init_transforms()

        # Pre-allocate dummy tensors to avoid CUDA memory reallocation per batch
        self._dummy_audio = torch.zeros(1, self.sample_rate, device=self.device)
        self._dummy_mel = torch.zeros(1, self.n_mels, 100, device=self.device)

    # ==========================================================
    # 🔧 TRANSFORM INITIALIZATION
    # ==========================================================
    def _init_transforms(self):
        """Initialize and cache torchaudio transforms."""
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            power=2.0,
            normalized=True,
        ).to(self.device)

        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "n_mels": self.n_mels,
                "f_min": self.fmin,
                "f_max": self.fmax,
            },
        ).to(self.device)

        self.inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=self.n_fft // 2 + 1,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
        ).to(self.device)

        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_iter=32,
            power=2.0,
        ).to(self.device)

    # ==========================================================
    # 🎵 MEL SPECTROGRAM
    # ==========================================================
    @torch.inference_mode()
    def extract_mel(self, audio: torch.Tensor, return_db: bool = True) -> torch.Tensor:
        """Compute Mel spectrogram (with GPU acceleration and NaN safety)."""
        try:
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()

            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

            if audio.numel() == 0 or torch.all(audio == 0):
                return self._dummy_mel.clone().cpu()

            # Non-blocking GPU transfer
            audio = audio.to(self.device, non_blocking=True)

            # Compute mel
            mel = self.mel_transform(audio)

            # Convert to decibels for perceptual scaling
            if return_db:
                mel = torchaudio.functional.amplitude_to_DB(
                    mel, multiplier=10.0, amin=1e-10, db_multiplier=0.0
                )

            mel = torch.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)
            return mel.cpu()

        except Exception as e:
            print(f"[ERROR] extract_mel failed: {e}")
            return self._dummy_mel.clone().cpu()

    # ==========================================================
    # 🎵 MFCC EXTRACTION
    # ==========================================================
    @torch.inference_mode()
    def extract_mfcc(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute MFCCs efficiently with GPU fallback."""
        try:
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()

            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

            if audio.numel() == 0:
                return torch.zeros((1, self.n_mfcc, 100))

            audio = audio.to(self.device, non_blocking=True)
            mfcc = self.mfcc_transform(audio)
            mfcc = torch.nan_to_num(mfcc, nan=0.0, posinf=0.0, neginf=0.0)
            return mfcc.cpu()

        except Exception as e:
            print(f"[ERROR] extract_mfcc failed: {e}")
            return torch.zeros((1, self.n_mfcc, 100))

    # ==========================================================
    # 🔄 MEL → AUDIO
    # ==========================================================
    @torch.inference_mode()
    def mel_to_audio(self, mel: torch.Tensor, use_griffin_lim: bool = True) -> torch.Tensor:
        """Convert mel back to audio using Griffin-Lim or librosa inverse."""
        try:
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)

            mel = mel.to(self.device)
            mel = torchaudio.functional.DB_to_amplitude(mel, ref=1.0, power=0.5)
            mel = torch.nan_to_num(mel)

            if use_griffin_lim:
                spec = self.inverse_mel(mel)
                audio = self.griffin_lim(spec)
            else:
                mel_np = mel.squeeze(0).cpu().numpy()
                audio_np = librosa.feature.inverse.mel_to_audio(
                    mel_np,
                    sr=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                )
                audio = torch.from_numpy(audio_np).float().unsqueeze(0)

            audio = torch.clamp(audio, -1.0, 1.0)
            return audio.cpu()

        except Exception as e:
            print(f"[WARN] mel_to_audio failed: {e}")
            return self._dummy_audio.cpu()

    # ==========================================================
    # 📈 STFT
    # ==========================================================
    @torch.inference_mode()
    def compute_stft(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute STFT magnitude + phase."""
        try:
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()

            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

            if audio.numel() == 0:
                raise ValueError("Empty input")

            audio = audio.to(self.device, non_blocking=True)
            stft = torch.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                return_complex=True,
            )
            mag = torch.abs(stft)
            phase = torch.angle(stft)
            return mag.cpu(), phase.cpu()

        except Exception as e:
            print(f"[ERROR] compute_stft failed: {e}")
            dummy = torch.zeros((1, self.n_fft // 2 + 1, 10))
            return dummy, dummy

    # ==========================================================
    # ⚖️ NORMALIZATION
    # ==========================================================
    @staticmethod
    def normalize_mel(mel: torch.Tensor, mean: Optional[float] = None, std: Optional[float] = None) -> torch.Tensor:
        mel = mel.to(torch.float32)
        if mel.numel() == 0:
            return mel
        mean = mel.mean() if mean is None else mean
        std = mel.std() + 1e-8 if std is None or std == 0 else std
        return torch.nan_to_num((mel - mean) / std)

    @staticmethod
    def denormalize_mel(mel: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        mel = mel.to(torch.float32)
        return torch.nan_to_num(mel * std + mean)
