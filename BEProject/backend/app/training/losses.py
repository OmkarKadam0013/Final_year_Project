# backend/app/training/losses.py
import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math

# ---------------------------
# Helpers
# ---------------------------
def _center_crop_time(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Center-crop two tensors to the same last-dimension length.
    Works for tensors of shape (..., T).
    """
    if a is None or b is None:
        raise ValueError("Input tensors must not be None")

    ta = a.size(-1)
    tb = b.size(-1)
    if ta == tb:
        return a, b

    m = min(ta, tb)
    # center indices
    def crop(x, m):
        t = x.size(-1)
        start = max(0, (t - m) // 2)
        return x[..., start:start + m]

    return crop(a, m), crop(b, m)


# ---------------------------
# GAN losses (LSGAN)
# ---------------------------
class GANLoss:
    """GAN losses for generator and discriminator (LSGAN)."""

    @staticmethod
    def generator_loss(disc_outputs_fake: List[torch.Tensor]) -> torch.Tensor:
        loss = 0.0
        count = 0
        for out in disc_outputs_fake:
            try:
                loss = loss + torch.mean((out - 1) ** 2)
                count += 1
            except Exception:
                # Skip invalid entries
                continue
        if count == 0:
            return torch.tensor(0.0, requires_grad=True)
        return loss / count

    @staticmethod
    def discriminator_loss(disc_outputs_real: List[torch.Tensor], disc_outputs_fake: List[torch.Tensor]) -> torch.Tensor:
        loss = 0.0
        count = 0
        for dr, dg in zip(disc_outputs_real, disc_outputs_fake):
            try:
                r_loss = torch.mean((dr - 1) ** 2)
                g_loss = torch.mean(dg ** 2)
                loss = loss + (r_loss + g_loss)
                count += 1
            except Exception:
                continue
        if count == 0:
            return torch.tensor(0.0, requires_grad=True)
        return loss / count


# ---------------------------
# Feature matching loss
# ---------------------------
class FeatureMatchingLoss:
    """
    Feature matching loss robust to different temporal lengths produced by discriminators.
    Expects:
       features_real: List (num_scales) of lists (layers) of tensors (B, C, T_i)
       features_fake: same structure
    """

    @staticmethod
    def forward(features_real: List[List[torch.Tensor]], features_fake: List[List[torch.Tensor]]) -> torch.Tensor:
        total = 0.0
        matched = 0

        # Safety: ensure both sequences have same number of scales (zip will stop at shortest)
        for feat_real_list, feat_fake_list in zip(features_real, features_fake):
            # iterate over layers within scale
            for feat_real, feat_fake in zip(feat_real_list, feat_fake_list):
                try:
                    # ensure tensors and numeric
                    if feat_real is None or feat_fake is None:
                        continue
                    # move to same device/dtype if needed
                    if feat_real.dtype != feat_fake.dtype:
                        feat_fake = feat_fake.to(feat_real.dtype)
                    # Align in time (last dim)
                    if feat_real.dim() >= 1 and feat_fake.dim() >= 1:
                        # If both have a time dimension
                        if feat_real.size(-1) != feat_fake.size(-1):
                            feat_fake, feat_real = _center_crop_time(feat_fake, feat_real)
                    # compute l1
                    total = total + F.l1_loss(feat_fake, feat_real.detach())
                    matched += 1
                except Exception:
                    # skip this pair
                    continue

        if matched == 0:
            return torch.tensor(0.0, requires_grad=True)
        return total / matched


# ---------------------------
# Mel reconstruction loss
# ---------------------------
class MelLoss:
    """Mel spectrogram reconstruction loss that tolerates small mismatches."""

    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80, device: Optional[torch.device] = None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            n_mels=n_mels
        ).to(self.device)

    def forward(self, audio_fake: torch.Tensor, audio_real: torch.Tensor) -> torch.Tensor:
        try:
            # expect shape (B, 1, T) or (B, T)
            def ensure_batch(a):
                if a.dim() == 2:  # (B, T) -> (B, 1, T)
                    return a.unsqueeze(1)
                return a

            af = ensure_batch(audio_fake).to(self.device)
            ar = ensure_batch(audio_real).to(self.device)

            # compute mel (returns shape (B, n_mels, time))
            mel_fake = self.mel_transform(af.squeeze(1) if af.size(1) == 1 else af.view(af.size(0), -1)).unsqueeze(1) if False else self.mel_transform(af.squeeze(1))
            mel_real = self.mel_transform(ar.squeeze(1) if ar.size(1) == 1 else ar.view(ar.size(0), -1))

            # align time dimension if needed
            if mel_fake.size(-1) != mel_real.size(-1):
                mel_fake, mel_real = _center_crop_time(mel_fake, mel_real)

            return F.l1_loss(mel_fake, mel_real)
        except Exception as e:
            print(f"[WARN] MelLoss failed: {e}")
            return torch.tensor(0.0, requires_grad=True)


# ---------------------------
# Multi-resolution STFT loss
# ---------------------------
class MultiResolutionSTFTLoss:
    """Multi-resolution STFT loss for audio."""

    def __init__(self, fft_sizes=[1024, 2048, 512],
                 hop_sizes=[256, 512, 128],
                 win_lengths=[1024, 2048, 512],
                 device: Optional[torch.device] = None):
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def stft_mag(self, x: torch.Tensor, fft_size: int, hop_size: int, win_length: int) -> torch.Tensor:
        """
        Return magnitude of STFT. x expected shape (B, 1, T) or (B, T).
        """
        if x is None:
            raise ValueError("Input is None")
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.dim() == 2:
            # (B, T)
            pass
        # ensure device
        x = x.to(self.device)
        win = torch.hann_window(win_length).to(self.device)
        stft = torch.stft(x, n_fft=fft_size, hop_length=hop_size, win_length=win_length,
                          window=win, return_complex=True)
        return torch.abs(stft)

    def forward(self, audio_fake: torch.Tensor, audio_real: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        count = 0
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            try:
                mag_f = self.stft_mag(audio_fake, fft_size, hop_size, win_length)
                mag_r = self.stft_mag(audio_real, fft_size, hop_size, win_length)

                # align time dims: (B, F, T)
                if mag_f.size(-1) != mag_r.size(-1):
                    mag_f, mag_r = _center_crop_time(mag_f, mag_r)

                # spectral convergence
                denom = torch.norm(mag_r, p='fro') + 1e-8
                sc_loss = torch.norm(mag_f - mag_r, p='fro') / denom

                # log magnitude loss
                log_f = torch.log(mag_f + 1e-7)
                log_r = torch.log(mag_r + 1e-7)
                log_loss = F.l1_loss(log_f, log_r)

                loss = loss + (sc_loss + log_loss)
                count += 1
            except Exception:
                continue

        if count == 0:
            return torch.tensor(0.0, requires_grad=True)
        return loss / count


# ---------------------------
# Other simple losses (wrappers)
# ---------------------------
class CycleLoss:
    @staticmethod
    def forward(reconstructed, original):
        try:
            # align in time if needed (last dim)
            if reconstructed.size(-1) != original.size(-1):
                reconstructed, original = _center_crop_time(reconstructed, original)
            return F.l1_loss(reconstructed, original)
        except Exception:
            return torch.tensor(0.0, requires_grad=True)


class PPGLoss:
    @staticmethod
    def forward(ppg_converted, ppg_original):
        try:
            if ppg_converted.size(-1) != ppg_original.size(-1):
                ppg_converted, ppg_original = _center_crop_time(ppg_converted, ppg_original)
            return F.l1_loss(ppg_converted, ppg_original)
        except Exception:
            return torch.tensor(0.0, requires_grad=True)


class SpeakerLoss:
    @staticmethod
    def forward(emb_converted, emb_original):
        try:
            cos_sim = F.cosine_similarity(emb_converted, emb_original, dim=1)
            return (1 - cos_sim).mean()
        except Exception:
            return torch.tensor(0.0, requires_grad=True)


# ---------------------------
# CombinedLoss
# ---------------------------
class CombinedLoss:
    """Combined loss managing all components while being fault-tolerant."""

    def __init__(self, config):
        self.config = config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # instantiate components
        self.gan_loss = GANLoss()
        self.feature_matching_loss = FeatureMatchingLoss()
        self.mel_loss = MelLoss(
            config.audio.sample_rate,
            config.audio.n_fft,
            config.audio.hop_length,
            config.audio.n_mels,
            device=device
        )
        self.stft_loss = MultiResolutionSTFTLoss(device=device)
        self.cycle_loss = CycleLoss()
        self.ppg_loss = PPGLoss()
        self.speaker_loss = SpeakerLoss()

    def compute_generator_loss(self,
                              disc_outputs_fake,
                              features_real,
                              features_fake,
                              mel_fake, mel_real,
                              audio_fake, audio_real,
                              ppg_fake, ppg_real,
                              emb_fake, emb_real,
                              reconstructed, original):
        """
        Compute generator loss using guarded calls. Any failing sub-loss becomes zero to avoid crash.
        Returns: (total_loss, dict_of_components)
        """

        # adversarial
        try:
            loss_adv = self.gan_loss.generator_loss(disc_outputs_fake)
        except Exception as e:
            print(f"[WARN] GAN loss failed: {e}")
            loss_adv = torch.tensor(0.0, requires_grad=True)

        # feature matching
        try:
            loss_fm = self.feature_matching_loss.forward(features_real, features_fake)
        except Exception as e:
            print(f"[WARN] Feature matching failed: {e}")
            loss_fm = torch.tensor(0.0, requires_grad=True)

        # mel
        try:
            loss_mel = self.mel_loss.forward(audio_fake, audio_real)
        except Exception as e:
            print(f"[WARN] Mel loss failed: {e}")
            loss_mel = torch.tensor(0.0, requires_grad=True)

        # stft
        try:
            loss_stft = self.stft_loss.forward(audio_fake, audio_real)
        except Exception as e:
            print(f"[WARN] STFT loss failed: {e}")
            loss_stft = torch.tensor(0.0, requires_grad=True)

        # cycle
        try:
            loss_cycle = self.cycle_loss.forward(reconstructed, original)
        except Exception as e:
            print(f"[WARN] Cycle loss failed: {e}")
            loss_cycle = torch.tensor(0.0, requires_grad=True)

        # ppg
        try:
            loss_ppg = self.ppg_loss.forward(ppg_fake, ppg_real)
        except Exception as e:
            print(f"[WARN] PPG loss failed: {e}")
            loss_ppg = torch.tensor(0.0, requires_grad=True)

        # speaker
        try:
            loss_speaker = self.speaker_loss.forward(emb_fake, emb_real)
        except Exception as e:
            print(f"[WARN] Speaker loss failed: {e}")
            loss_speaker = torch.tensor(0.0, requires_grad=True)

        # Compose total using config lambdas (fallback safe values if missing)
        def _get(cfg_name, default):
            return getattr(self.config.training, cfg_name, default) if hasattr(self.config, "training") else default

        lambda_gan = _get("lambda_gan", 1.0)
        lambda_feat = _get("lambda_feat_match", 10.0)
        lambda_mel = _get("lambda_mel", 45.0)
        lambda_cycle = _get("lambda_cycle", 1.0)
        lambda_ppg = _get("lambda_ppg", 1.0)
        lambda_speaker = _get("lambda_speaker", 1.0)

        total_loss = (
            lambda_gan * loss_adv +
            lambda_feat * loss_fm +
            lambda_mel * (loss_mel + loss_stft) +
            lambda_cycle * loss_cycle +
            lambda_ppg * loss_ppg +
            lambda_speaker * loss_speaker
        )

        losses = {
            'total': total_loss,
            'adv': loss_adv,
            'fm': loss_fm,
            'mel': loss_mel,
            'stft': loss_stft,
            'cycle': loss_cycle,
            'ppg': loss_ppg,
            'speaker': loss_speaker
        }

        return total_loss, losses

    def compute_discriminator_loss(self, disc_outputs_real, disc_outputs_fake):
        try:
            return self.gan_loss.discriminator_loss(disc_outputs_real, disc_outputs_fake)
        except Exception as e:
            print(f"[WARN] Discriminator loss failed: {e}")
            return torch.tensor(0.0, requires_grad=True)
