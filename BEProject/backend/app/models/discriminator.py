import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeInstanceNorm1d(nn.Module):
    """
    Wrapper around InstanceNorm1d that safely handles small spatial sizes.
    Falls back to GroupNorm when batch size is 1 or time dimension is tiny.
    GroupNorm works fine for batch=1.
    """
    def __init__(self, num_features):
        super().__init__()
        # instance norm with affine parameters (normal behavior)
        self.inorm = nn.InstanceNorm1d(num_features, affine=True, eps=1e-5)
        # group norm fallback (works with batch size 1)
        # using 1 group -> behaves like LayerNorm/InstanceNorm across channels
        self.gnorm = nn.GroupNorm(1, num_features, eps=1e-5)
        self.num_features = num_features

    def forward(self, x):
        # x shape: (B, C, T)
        B = x.size(0)
        T = x.size(-1)
        # If time dimension is too small or batch is 1, prefer GroupNorm
        if B == 1 or T <= 1:
            try:
                return self.gnorm(x)
            except Exception:
                # Last resort: identity (no norm) to avoid crash
                return x
        try:
            return self.inorm(x)
        except Exception:
            # If InstanceNorm fails, fallback to GroupNorm
            try:
                return self.gnorm(x)
            except Exception:
                return x


class DiscriminatorBlock(nn.Module):
    """Single 1D convolution block with normalization and activation."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = SafeInstanceNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if x.size(-1) == 0:  # avoid collapse due to stride/padding
            x = x.unsqueeze(-1)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ScaleDiscriminator(nn.Module):
    """
    Single-scale discriminator — captures local temporal dependencies in mel-spectrograms.
    """
    def __init__(self, input_channels=80, base_channels=64):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_channels, base_channels, 15, 1, 7),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            DiscriminatorBlock(base_channels, base_channels * 2, 41, 4, 20),
            DiscriminatorBlock(base_channels * 2, base_channels * 4, 41, 4, 20),
            DiscriminatorBlock(base_channels * 4, base_channels * 8, 41, 4, 20),
            DiscriminatorBlock(base_channels * 8, base_channels * 8, 41, 4, 20),
            DiscriminatorBlock(base_channels * 8, base_channels * 8, 5, 1, 2),
        ])

        self.output = nn.Conv1d(base_channels * 8, 1, 3, 1, 1)

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        x = self.output(x)
        features.append(x)
        return x, features


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator — processes mel-spectrograms at multiple temporal resolutions.
    """
    def __init__(self, config, num_scales=3):
        super().__init__()
        self.config = config

        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(config.audio.n_mels, config.model.discriminator_channels)
            for _ in range(num_scales)
        ])

        # Safe average pooling for downsampling
        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        """
        Args:
            x: Tensor (B, n_mels, T)
        Returns:
            outputs: list of discriminator outputs
            all_features: list of intermediate features for feature matching loss
        """
        outputs, all_features = [], []

        for i, discriminator in enumerate(self.discriminators):
            if i > 0:
                if x.size(-1) < 4:  # avoid too short sequences
                    break
                x = self.downsample(x)

            out, features = discriminator(x)
            outputs.append(out)
            all_features.append(features)

        return outputs, all_features


class PeriodDiscriminator(nn.Module):
    """
    Period discriminator — reshapes waveform to (B, C, T//period, period)
    to capture periodic structure at a given period.
    """
    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period

        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(2, 0)),
            nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0)),
            nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(2, 0)),
            nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(2, 0)),
            nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)),
        ])

        self.output = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape

        # Pad time dimension so that it is divisible by period
        if T % self.period != 0:
            pad_len = self.period - (T % self.period)
            x = F.pad(x, (0, pad_len), mode='reflect')
            T += pad_len

        # Reshape to (B, 1, C*T//period, period)
        x = x.view(B, 1, C * (T // self.period), self.period)

        features = []
        for conv in self.convs:
            x = conv(x)
            x = self.activation(x)
            features.append(x)

        x = self.output(x)
        features.append(x)

        return x, features


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-period discriminator for capturing periodicity across various resolutions.
    Typically used alongside MultiScaleDiscriminator.
    """
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(period) for period in periods
        ])

    def forward(self, x):
        outputs, all_features = [], []

        for disc in self.discriminators:
            out, features = disc(x)
            outputs.append(out)
            all_features.append(features)

        return outputs, all_features
