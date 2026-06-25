import torch
import torch.nn as nn
import math


class RibeiroResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=16, stride=1, dropout_rate=0.2):
        super().__init__()
        self.stride = stride
        self.needs_projection = (stride > 1) or (in_channels != out_channels)

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=stride,
            bias=False
        )

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.shortcut = nn.Identity()
        if self.needs_projection:
            self.shortcut = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            )

    @staticmethod
    def _match_length(a: torch.Tensor, b: torch.Tensor):
        # Center-crop to the minimum length so residual add always works
        la, lb = a.shape[-1], b.shape[-1]
        if la == lb:
            return a, b
        m = min(la, lb)

        def crop(x):
            l = x.shape[-1]
            start = (l - m) // 2
            return x[..., start:start + m]

        return crop(a), crop(b)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.bn1(x)
        out = self.act1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv2(out)
        out = self.dropout(out)

        out, identity = self._match_length(out, identity)
        return out + identity


class RibeiroECGNet(nn.Module):
    """
    1-lead variable-length version inspired by Ribeiro et al. ResNet for ECG
    Designed for 500 Hz signals — you pass desired duration in minutes

    Example usage:
        model = RibeiroECGNet(
            n_classes=1,           # binary or change to your number
            duration_minutes=10.0, # target / reference duration
            fs=500,
            kernel_size=16
        )
    """

    def __init__(
        self,
        n_classes: int = 1,
        duration_minutes: float = 10.0,
        fs: int = 500,
        kernel_size: int = 16,
        dropout_rate: float = 0.2,
        final_dropout: float = 0.5,
    ):
        super().__init__()

        self.fs = fs
        self.target_length = int(
            duration_minutes * 60 * fs
        )  # e.g. 10 min → 300,000 samples

        # Initial stem — stronger downsampling at beginning
        self.initial_conv = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=kernel_size,
            stride=2,  # mild initial downsampling
            padding=kernel_size // 2,
            bias=False,
        )
        self.initial_bn = nn.BatchNorm1d(64)
        self.initial_act = nn.ReLU(inplace=True)

        # Residual stages — similar progression to Ribeiro
        # We gradually reduce temporal dimension
        self.stage1 = self._make_stage(
            64, 128, stride=4, n_blocks=1, kernel_size=kernel_size, dropout=dropout_rate
        )
        self.stage2 = self._make_stage(
            128,
            196,
            stride=4,
            n_blocks=1,
            kernel_size=kernel_size,
            dropout=dropout_rate,
        )
        self.stage3 = self._make_stage(
            196,
            256,
            stride=4,
            n_blocks=1,
            kernel_size=kernel_size,
            dropout=dropout_rate,
        )
        self.stage4 = self._make_stage(
            256,
            320,
            stride=4,
            n_blocks=1,
            kernel_size=kernel_size,
            dropout=dropout_rate,
        )

        # Final
        self.final_bn = nn.BatchNorm1d(320)
        self.final_act = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # ← key for variable length
        self.dropout = nn.Dropout(final_dropout)
        self.head = nn.Linear(320, n_classes)

    def _make_stage(self, in_ch, out_ch, stride, n_blocks, kernel_size, dropout):
        layers = []
        # First block may downsample
        layers.append(
            RibeiroResidualBlock(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                dropout_rate=dropout,
            )
        )
        # Additional blocks at same resolution
        for _ in range(1, n_blocks):
            layers.append(
                RibeiroResidualBlock(
                    out_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    dropout_rate=dropout,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (B, 1, L) or (B, L) — any reasonable length
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, L) → (B, 1, L)

        # Optional: truncate / pad to reference length (you can also remove this)
        if self.target_length is not None:
            curr_len = x.shape[-1]
            if curr_len > self.target_length:
                start = (curr_len - self.target_length) // 2
                x = x[..., start : start + self.target_length]
            elif curr_len < self.target_length:
                pad_left = (self.target_length - curr_len) // 2
                pad_right = self.target_length - curr_len - pad_left
                x = nn.functional.pad(
                    x, (pad_left, pad_right), mode="constant", value=0
                )

        # Forward
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_act(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.final_bn(x)
        x = self.final_act(x)

        # Global average pooling → fixed size regardless of input length
        x = self.global_pool(x).squeeze(-1)  # (B, 320)
        x = self.dropout(x)
        logits = self.head(x)  # (B, n_classes)

        return logits


# ────────────────────────────────────────────────
# Example instantiation & summary
# ────────────────────────────────────────────────
# if __name__ == "__main__":
# import torchsummary

"""
model = RibeiroECGNet(
    n_classes=1,  # ← change to your task (e.g. 6)
    duration_minutes=10.0,  # ← main parameter you wanted
    fs=500,
    kernel_size=16,
    dropout_rate=0.2,
    final_dropout=0.5,
)

# Test with different lengths
x1 = torch.randn(2, 1, 900_000)  # 15 min @ 500 Hz
x2 = torch.randn(2, 300_000)  # 10 min @ 500 Hz (will be used as-is or centered)

print("Output shape 15 min:", model(x1).shape)
print("Output shape 10 min:", model(x2).shape)

# Optional: summary (install torchsummary if needed)
# torchsummary.summary(model, input_size=(1, 500*60*10))  # 10 min example
"""
