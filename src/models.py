import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Q1 Models: MLP Variants
# ==============================================================================

class DeepMLP(nn.Module):
    """
    Standard Deep Global MLP with batch normalization and ReLU activations.
    Input: (B, 3, 32, 32) flattened to (B, 3072).
    Architecture: 3072 -> 160 -> 128 -> 64 -> 10.
    Parameter count: ~521k parameters.
    """
    def __init__(self, input_dim=3072, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(inplace=True),
            nn.Linear(160, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class ShallowWideMLP(nn.Module):
    """
    Wide Shallow Global MLP with parameter budget matched to DeepMLP (~518k parameters).
    Architecture: 3072 -> 168 -> 10.
    """
    def __init__(self, input_dim=3072, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 168),
            nn.BatchNorm1d(168),
            nn.ReLU(inplace=True),
            nn.Linear(168, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class LocallyConnectedPatchMLP(nn.Module):
    """
    Locally-Connected Patch MLP without weight sharing across spatial positions.
    Image is divided into sixteen 8x8 patches (16 patches * 3 channels * 8 * 8 = 16 * 192).
    Each patch has an INDEPENDENT linear projection (no convolutional weight sharing).
    Followed by dense mixing layer and classification head.
    Parameter count matched to ~520k parameters.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # 16 independent linear layers: 192 -> 64 (16 * (192*64 + 64) = 197,632 params)
        self.patch_linears = nn.ModuleList([
            nn.Linear(192, 64) for _ in range(16)
        ])
        self.patch_bns = nn.ModuleList([
            nn.BatchNorm1d(64) for _ in range(16)
        ])
        # After concatenating 16 * 64 = 1024 features:
        # Linear(1024, 300) -> 307,500 params
        # Linear(300, 10) -> 3,010 params
        # Total params: 197,632 + 307,500 + 3,010 + BN params = ~510k params!
        self.fc1 = nn.Linear(1024, 300)
        self.bn1 = nn.BatchNorm1d(300)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(300, num_classes)

    def forward(self, x):
        # x: (B, 3, 32, 32)
        B = x.size(0)
        # Extract 16 patches of 8x8
        # Unfold spatial dimensions: (B, 3, 4, 8, 4, 8)
        patches = x.view(B, 3, 4, 8, 4, 8)
        # Permute to (B, 4, 4, 3, 8, 8) -> (B, 16, 192)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous().view(B, 16, 192)

        out_patches = []
        for i in range(16):
            p = patches[:, i, :]
            p_out = self.relu(self.patch_bns[i](self.patch_linears[i](p)))
            out_patches.append(p_out)

        h = torch.cat(out_patches, dim=1) # (B, 1024)
        h = self.relu(self.bn1(self.fc1(h)))
        out = self.fc2(h)
        return out


# ==============================================================================
# Q2 Models: CNN Classifier & Feature Extractor
# ==============================================================================

class ConvClassifier(nn.Module):
    """
    Standard CNN with parameter count closely matched to MLP (~570k parameters).
    Architecture:
      Conv(3, 32, 3, p=1) -> BN -> ReLU
      Conv(32, 64, 3, p=1) -> BN -> ReLU -> MaxPool(2) [16x16]
      Conv(64, 128, 3, p=1) -> BN -> ReLU
      Conv(128, 128, 3, p=1) -> BN -> ReLU -> MaxPool(2) [8x8]
      Conv(128, 256, 3, p=1) -> BN -> ReLU
      AdaptiveAvgPool(1) -> 256
      FC(256, 128) -> ReLU -> FC(128, 10)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, num_classes)

    def extract_features(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1)
        feat = self.relu(self.bn_fc(self.fc1(h)))
        return feat

    def forward(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1)
        h = self.relu(self.bn_fc(self.fc1(h)))
        out = self.fc2(h)
        return out


# ==============================================================================
# Q3 Models: Convolutional Autoencoders (3 Bottlenecks) & Denoising AE
# ==============================================================================

class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder with variable bottleneck dimension.
    Encoder:
      Conv(3, 32, 3, stride=2, padding=1) [16x16] -> ReLU
      Conv(32, 64, 3, stride=2, padding=1) [8x8] -> ReLU
      Conv(64, 128, 3, stride=2, padding=1) [4x4] -> ReLU
      Flatten -> Linear(2048, bottleneck_dim)
    Decoder:
      Linear(bottleneck_dim, 2048) -> ReLU -> Unflatten(128, 4, 4)
      ConvTranspose2d(128, 64, 4, stride=2, padding=1) [8x8] -> ReLU
      ConvTranspose2d(64, 32, 4, stride=2, padding=1) [16x16] -> ReLU
      ConvTranspose2d(32, 3, 4, stride=2, padding=1) [32x32] -> Sigmoid (or linear if normalized)
    """
    def __init__(self, bottleneck_dim=256, normalize_output=True):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.normalize_output = normalize_output

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.fc_enc = nn.Linear(128 * 4 * 4, bottleneck_dim)

        # Decoder
        self.fc_dec = nn.Linear(bottleneck_dim, 128 * 4 * 4)
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        z = self.fc_enc(h)
        return z

    def decode(self, z):
        h = F.relu(self.fc_dec(z))
        h = h.view(h.size(0), 128, 4, 4)
        rec = self.decoder_deconv(h)
        return rec

    def forward(self, x):
        z = self.encode(x)
        rec = self.decode(z)
        return rec


class LinearProbeClassifier(nn.Module):
    """Linear probe classifier trained on frozen autoencoder representations."""
    def __init__(self, in_features, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


def count_parameters(model):
    """Returns total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
