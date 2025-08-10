import torch
import torch.nn as nn
import torch.nn.functional as F


class OpticalDecoder(nn.Module):
    
    def __init__(self, input_channels=1, output_channels=1, features=(64, 128, 256)):
        super().__init__()
        self.features = features
        
        # Encoder path
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_channels = input_channels
        for feature_dim in features:
            self.encoders.append(self._make_encoder_block(in_channels, feature_dim))
            self.pools.append(nn.MaxPool2d(2))
            in_channels = feature_dim
        
        # Bottleneck
        self.bottleneck = self._make_encoder_block(features[-1], features[-1] * 2)
        
        # Decoder path
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i in range(len(features) - 1, -1, -1):
            in_feat = features[-1] * 2 if i == len(features) - 1 else features[i + 1]
            out_feat = features[i]
            
            self.upconvs.append(nn.ConvTranspose2d(in_feat, out_feat, 2, stride=2))
            self.decoders.append(self._make_decoder_block(out_feat * 2, out_feat))
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], output_channels, 1)
        self.output_activation = nn.Sigmoid()
    
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return self._make_encoder_block(in_channels, out_channels)
    
    def forward(self, x):
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Normalize input
        x = x / (x.max() + 1e-8)
        
        # Encoder path with skip connections
        encoder_features = []
        
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            encoder_features.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            
            # Skip connection
            encoder_feat = encoder_features[-(i + 1)]
            
            # Handle size mismatch
            if x.shape[2:] != encoder_feat.shape[2:]:
                x = F.interpolate(x, size=encoder_feat.shape[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, encoder_feat], dim=1)
            x = decoder(x)
        
        # Final convolution
        x = self.final_conv(x)
        x = self.output_activation(x)
        
        # Remove channel dimension if it was added
        if x.shape[1] == 1:
            x = x.squeeze(1)
        
        return x


class LightweightDecoder(nn.Module):
    
    def __init__(self, input_channels=1, output_channels=1, base_features=32):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, base_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features, base_features * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features * 2, base_features * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_features * 4, base_features * 2, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_features * 2, base_features, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features, output_channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x / (x.max() + 1e-8)
        x = self.encoder(x)
        x = self.decoder(x)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        return x