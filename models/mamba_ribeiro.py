import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional dependency:
# pip install mamba-ssm
try:
    from mamba_ssm.modules.mamba_simple import Mamba
except Exception:
    Mamba = None


class RibeiroResidualBlock(nn.Module):
    """
    Your conv residual block with length-matching to avoid off-by-one issues
    under strided convs / even kernels.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=16, stride=1, dropout_rate=0.2
    ):
        super().__init__()
        self.needs_projection = (stride > 1) or (in_channels != out_channels)

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=stride,
            bias=False,
        )

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.shortcut = nn.Identity()
        if self.needs_projection:
            self.shortcut = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )

    @staticmethod
    def _match_length(a: torch.Tensor, b: torch.Tensor):
        la, lb = a.shape[-1], b.shape[-1]
        if la == lb:
            return a, b
        m = min(la, lb)

        def crop(x):
            l = x.shape[-1]
            start = (l - m) // 2
            return x[..., start : start + m]

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


class MambaResidualBlock(nn.Module):
    """
    Residual Mamba block for sequences:
      input:  (B, C, L)
      mamba expects (B, L, C)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "mamba-ssm is not installed. Run: pip install mamba-ssm "
                "or ask me for a dependency-free alternative."
            )

        self.norm = nn.LayerNorm(d_model)  # works on (B, L, C)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, L) -> (B, L, C)
        x_seq = x.transpose(1, 2)
        y = self.norm(x_seq)
        y = self.mamba(y)
        y = self.drop(y)
        y = y + x_seq
        # back to (B, C, L)
        return y.transpose(1, 2)


class Downsample1d(nn.Module):
    """
    Simple downsampling module to reduce L before Mamba for speed.
    Uses strided conv to keep it learnable and avoid pooling quirks.
    """

    def __init__(self, channels: int, stride: int):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RibeiroMambaECGNet(nn.Module):
    """
    Hybrid: Conv-ResNet stem/stages -> downsample -> Mamba blocks -> global pool -> head

    Idea:
    - Early conv stages: morphology (P/QRS/T) + robust feature extraction.
    - Later Mamba: long-range temporal structure (rhythm, irregularity) at reduced length.

    Input:
      x: (B, 1, L) or (B, L)
    """

    def __init__(
        self,
        n_classes: int = 1,
        duration_minutes: float | None = 10.0,
        fs: int = 500,
        kernel_size: int = 16,
        conv_dropout: float = 0.2,
        mamba_dropout: float = 0.1,
        final_dropout: float = 0.5,
        # Mamba params
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        n_mamba_blocks: int = 4,
        # Extra downsample before Mamba for speed
        pre_mamba_stride: int = 4,
    ):
        super().__init__()
        self.fs = fs
        self.target_length = (
            None if duration_minutes is None else int(duration_minutes * 60 * fs)
        )

        # Stem
        self.initial_conv = nn.Conv1d(
            1,
            64,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            bias=False,
        )
        self.initial_bn = nn.BatchNorm1d(64)
        self.initial_act = nn.ReLU(inplace=True)

        # Conv residual stages (you can tune strides/widths)
        self.stage1 = RibeiroResidualBlock(
            64, 128, kernel_size=kernel_size, stride=4, dropout_rate=conv_dropout
        )
        self.stage2 = RibeiroResidualBlock(
            128, 196, kernel_size=kernel_size, stride=4, dropout_rate=conv_dropout
        )
        self.stage3 = RibeiroResidualBlock(
            196, 256, kernel_size=kernel_size, stride=4, dropout_rate=conv_dropout
        )

        # Before Mamba: fix channels to a "d_model" and downsample length
        self.to_d_model = nn.Conv1d(256, 256, kernel_size=1, bias=False)
        self.pre_mamba_down = Downsample1d(256, stride=pre_mamba_stride)

        # Mamba stack at lower L
        self.mamba_blocks = nn.Sequential(
            *[
                MambaResidualBlock(
                    d_model=256,
                    d_state=mamba_d_state,
                    d_conv=mamba_d_conv,
                    expand=mamba_expand,
                    dropout=mamba_dropout,
                )
                for _ in range(n_mamba_blocks)
            ]
        )

        # Head
        self.final_bn = nn.BatchNorm1d(256)
        self.final_act = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(final_dropout)
        self.head = nn.Linear(256, n_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Optional length normalize
        if self.target_length is not None:
            curr_len = x.shape[-1]
            if curr_len > self.target_length:
                start = (curr_len - self.target_length) // 2
                x = x[..., start : start + self.target_length]
            elif curr_len < self.target_length:
                pad_left = (self.target_length - curr_len) // 2
                pad_right = self.target_length - curr_len - pad_left
                x = F.pad(x, (pad_left, pad_right), mode="constant", value=0)

        x = self.initial_act(self.initial_bn(self.initial_conv(x)))

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.to_d_model(x)
        x = self.pre_mamba_down(x)

        x = self.mamba_blocks(x)

        x = self.final_act(self.final_bn(x))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.head(x)


"""
# if __name__ == "__main__":
model = RibeiroMambaECGNet(
    n_classes=1,
    duration_minutes=10.0,
    fs=500,
    kernel_size=16,
    n_mamba_blocks=4,
    pre_mamba_stride=4,
)
x1 = torch.randn(2, 1, 450_000)
x2 = torch.randn(2, 300_000)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

x1 = x1.to(device)
x2 = x2.to(device)

with torch.no_grad():
    y1 = model(x1)
    y2 = model(x2)
print(y1.shape, y2.shape)
"""
