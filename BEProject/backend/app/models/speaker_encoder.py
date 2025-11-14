# backend/app/models/speaker_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (B, C, T)
        squeeze = x.mean(dim=2)  # (B, C)
        excitation = self.fc(squeeze).unsqueeze(2)  # (B, C, 1)
        return x * excitation

class Res2NetBlock(nn.Module):
    """Res2Net block for multi-scale processing"""
    def __init__(self, channels, kernel_size, dilation, scale=8):
        super().__init__()
        self.scale = scale
        width = channels // scale
        
        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size, padding=dilation, dilation=dilation)
            for _ in range(scale-1)
        ])
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = []
        spx = torch.chunk(x, self.scale, dim=1)
        
        out.append(spx[0])
        for i in range(1, self.scale):
            if i == 1:
                sp = spx[i]
            else:
                sp = spx[i] + out[-1]
            sp = self.convs[i-1](sp)
            out.append(sp)
        
        out = torch.cat(out, dim=1)
        out = self.bn(out)
        out = self.relu(out + residual)
        
        return out

class SpeakerEncoder(nn.Module):
    """Advanced speaker encoder inspired by ECAPA-TDNN"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        channels = [512, 512, 512, 512, 1536]
        
        # Input layer
        self.conv1 = nn.Conv1d(config.audio.n_mels, channels[0], kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.relu = nn.ReLU()
        
        # Res2Net blocks with different dilations
        self.layer1 = Res2NetBlock(channels[0], kernel_size=3, dilation=2)
        self.layer2 = Res2NetBlock(channels[1], kernel_size=3, dilation=3)
        self.layer3 = Res2NetBlock(channels[2], kernel_size=3, dilation=4)
        
        # SE-Res2Net block
        self.se_layer = SEBlock(channels[2])
        
        # Aggregation
        self.conv2 = nn.Conv1d(channels[2], channels[4], kernel_size=1)
        
        # Attentive statistics pooling
        self.attention = nn.Sequential(
            nn.Conv1d(channels[4] * 3, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, channels[4], kernel_size=1),
            nn.Softmax(dim=2)
        )
        
        # Final embedding
        self.fc = nn.Linear(channels[4] * 2, config.model.speaker_emb_dim)
        self.bn_fc = nn.BatchNorm1d(config.model.speaker_emb_dim)
    
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, n_mels, T)
        Returns:
            embedding: (B, speaker_emb_dim)
        """
        # Initial convolution
        x = self.conv1(mel)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Res2Net blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.se_layer(x)
        
        # Aggregation
        x = self.conv2(x)
        
        # Attentive statistics pooling
        t = x.size(2)
        mean = x.mean(dim=2, keepdim=True).expand(-1, -1, t)
        std = x.std(dim=2, keepdim=True).expand(-1, -1, t)
        attn_input = torch.cat([x, mean, std], dim=1)
        
        alpha = self.attention(attn_input)
        mean = (alpha * x).sum(dim=2)
        std = torch.sqrt((alpha * (x ** 2)).sum(dim=2) - mean ** 2 + 1e-8)
        
        # Concatenate statistics
        stats = torch.cat([mean, std], dim=1)
        
        # Final embedding
        embedding = self.fc(stats)
        embedding = self.bn_fc(embedding)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
