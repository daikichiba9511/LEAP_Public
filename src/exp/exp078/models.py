import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Model
# =============================================================================


def x_to_seq(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: (batch_size, 556)

    Returns:
        x_seq: (batch_size, 60, 25)
    """
    # (N, 556) -> (N, 60, 6)
    x_seq0 = x[:, 0 : 6 * 60].reshape(-1, 6, 60).permute(0, 2, 1)
    # (N, 556) -> (N, 60, 3)
    x_seq1 = x[:, 6 * 60 + 16 : 9 * 60 + 16].reshape(-1, 3, 60).permute(0, 2, 1)
    # (N, 556) -> (N, 60, 1)
    x_flat = x[:, 6 * 60 : 6 * 60 + 16].reshape(-1, 1, 16).repeat(1, 60, 1)
    return torch.cat([x_seq0, x_seq1, x_flat], dim=-1)


class FFN(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.2, hidden_size: int | None = None):
        super().__init__()
        if hidden_size is None:
            hidden_size = out_features
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """

    Attributes:
        mha: MultiheadAttention
        ln1: LayerNorm
        ln2: LayerNorm
        seq: nn.Sequential

    Refs:
    [1]
    https://www.kaggle.com/code/baurzhanurazalinov/parkinson-s-freezing-tdcsfog-training-code
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        seq_model_dim: int = 320,
        encoder_dropout: float = 0.2,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=encoder_dropout,
            device=device,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(embed_dim, device=device)
        self.ln2 = nn.LayerNorm(embed_dim, device=device)
        self.ffn = FFN(embed_dim, seq_model_dim, dropout_rate=encoder_dropout).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.mha(query=x, value=x, key=x, need_weights=False)
        x = self.ln1(x + attn_out)
        x = x + self.ffn(x)
        x = self.ln2(x)
        return x


class GeM(nn.Module):
    def __init__(self, p: int = 3, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x: torch.Tensor, p: int | torch.Tensor = 3, eps: float = 1e-6) -> torch.Tensor:
        # return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)
        return F.avg_pool1d(x.clamp(min=eps).pow(p), x.size(-1)).pow(1.0 / p)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3, stride: int = 1, padding: int | str = "same") -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 256, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 128, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, kernel_size, stride, padding)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        return x


class TransformerEncoder(nn.Module):
    """

    Refs:
    [1]
    https://www.kaggle.com/code/baurzhanurazalinov/parkinson-s-freezing-tdcsfog-training-code
    """

    def __init__(
        self,
        model_dim: int = 320,
        dropout_rate: float = 0.2,
        num_encoder_layers: int = 3,
        embed_dim: int = 128,
        num_heads: int = 8,
        seq_model_dim: int = 320,
        seq_len: int = 3000,
        device: torch.device | None = None,
        bs: int = 24,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        self.bs = bs

        self.cnn = CNNBlock(in_channels=seq_len)

        self.fc1 = nn.Linear(model_dim, embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.pos_encoding = nn.Parameter(
            torch.normal(mean=0, std=0.02, size=(1, self.seq_len, embed_dim)).to(device),
            requires_grad=True,
        )
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                seq_model_dim=seq_model_dim,
                encoder_dropout=dropout_rate,
                device=device,
            )
            for _ in range(num_encoder_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (bs, seq_len, model_dim)

        Returns:
            x: (bs, seq_len, model_dim)
        """
        x = self.fc1(x)

        # add position encoding
        bs = x.shape[0]
        x = x + torch.tile(input=self.pos_encoding, dims=(bs, 1, 1))

        x = self.dropout1(x)

        x = self.cnn(x)

        # Ref:
        # [1]
        # https://www.kaggle.com/code/cdeotte/tensorflow-transformer-0-112/notebook#Build-Model
        for i in range(len(self.encoder_layers)):
            x_old = x
            x = self.encoder_layers[i](x)
            x = 0.7 * x + 0.3 * x_old  # skip connrection

        return x


class SEModule(nn.Module):
    def __init__(self, channel: int, reduction: int = 8) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.squeeze_ope = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channel, seq_len)
        """
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.squeeze_ope(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEResidualBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | str = "same",
        last_bn_act: bool = False,
    ) -> None:
        super().__init__()

        self.register_buffer("last_bn_act", torch.tensor(last_bn_act, dtype=torch.bool))

        self.bn0 = nn.BatchNorm1d(in_chan)
        self.conv1 = nn.Conv1d(in_chan, out_chan, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_chan)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_chan, out_chan, kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_chan)

        self.se = SEModule(out_chan)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.bn0(x)
        x = self.relu(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)

        x = self.se(x)

        x = x + skip
        if self.last_bn_act:
            x = self.bn2(x)
            x = self.relu(x)
        return x


def build_conv1d_block(
    in_chan: int, out_chan: int, kernel_size: int = 3, stride: int = 1, padding: int | str = "same"
) -> nn.Module:
    return nn.Sequential(*[
        nn.Conv1d(in_chan, out_chan, kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.BatchNorm1d(out_chan),
        nn.Conv1d(out_chan, out_chan, kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.BatchNorm1d(out_chan),
    ])


class SECNNResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int | str = "same"
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()

        self.res1 = SEResidualBlock(out_channels, out_channels, kernel_size=kernel_size, last_bn_act=False)
        self.res2 = SEResidualBlock(out_channels, out_channels, kernel_size=kernel_size, last_bn_act=True)

        self.res3 = SEResidualBlock(out_channels, out_channels, kernel_size=kernel_size, last_bn_act=False)
        self.res4 = SEResidualBlock(out_channels, out_channels, kernel_size=kernel_size, last_bn_act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)

        x = self.relu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.res3(x)
        x = self.res4(x)

        x = self.relu(x)
        x = self.bn2(x)

        return x


class UnetEncoderBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        pool_size: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | str = "same",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_chan, out_chan, kernel_size=kernel_size, stride=stride, padding=padding)
        self.res1 = SEResidualBlock(out_chan, out_chan, kernel_size=kernel_size, last_bn_act=False)
        self.res2 = SEResidualBlock(out_chan, out_chan, kernel_size=kernel_size, last_bn_act=True)
        self.pool = nn.AdaptiveAvgPool1d(pool_size // 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        x_skip = x
        x = self.res1(x)
        x = self.res2(x)
        x = x + x_skip
        p = self.pool(x)
        return x, p


class UnetDecoderBlock(nn.Module):
    def __init__(
        self, in_chan: int, out_chan: int, kernel_size: int = 3, stride: int = 1, padding: int | str = "same"
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(in_chan, out_chan, kernel_size=2, stride=2)
        self.conv = SECNNResidualBlock(in_chan, out_chan, kernel_size=kernel_size, stride=stride, padding=padding)
        self.res1 = SEResidualBlock(
            out_chan, out_chan, kernel_size=kernel_size, stride=stride, padding=padding, last_bn_act=False
        )
        self.res2 = SEResidualBlock(
            out_chan, out_chan, kernel_size=kernel_size, stride=stride, padding=padding, last_bn_act=True
        )

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.concat([x, feat], dim=1)
        x = self.conv(x)
        x_skip = x
        x = self.res1(x)
        x = self.res2(x)
        x = x + x_skip
        return x


class Unet1D(nn.Module):
    def __init__(
        self, in_chans: tuple[int, ...] = (25, 32, 64, 128, 256, 512), kernel_size: int = 3, stride: int = 1
    ) -> None:
        super().__init__()
        if len(in_chans) != 6:
            raise ValueError(f"Expected 6 in_chans, got {len(in_chans)}")

        self.encode0 = UnetEncoderBlock(
            in_chans[0], in_chans[1], kernel_size=kernel_size, stride=stride, padding="same", pool_size=64
        )  # in -> 32
        self.encode1 = UnetEncoderBlock(
            in_chans[1], in_chans[2], kernel_size=kernel_size, stride=stride, padding="same", pool_size=32
        )  # 32 -> 64
        self.encode2 = UnetEncoderBlock(
            in_chans[2], in_chans[3], kernel_size=kernel_size, stride=stride, padding="same", pool_size=16
        )  # 64 -> 128
        self.encode3 = UnetEncoderBlock(
            in_chans[3], in_chans[4], kernel_size=kernel_size, stride=stride, padding="same", pool_size=8
        )  # 128 -> 256

        self.middle = SECNNResidualBlock(
            in_chans[4], in_chans[5], kernel_size=kernel_size, stride=stride, padding="same"
        )  # 256 -> 512

        self.decode3 = UnetDecoderBlock(
            in_chans[5], in_chans[4], kernel_size=kernel_size, stride=stride, padding="same"
        )  # 512 -> 256
        self.decode2 = UnetDecoderBlock(
            in_chans[4], in_chans[3], kernel_size=kernel_size, stride=stride, padding="same"
        )  # 256 -> 128
        self.decode1 = UnetDecoderBlock(
            in_chans[3], in_chans[2], kernel_size=kernel_size, stride=stride, padding="same"
        )  # 128 -> 64
        self.decode0 = UnetDecoderBlock(
            in_chans[2], in_chans[1], kernel_size=kernel_size, stride=stride, padding="same"
        )  # 64 -> 32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0, p0 = self.encode0(x)
        x1, p1 = self.encode1(p0)
        x2, p2 = self.encode2(p1)
        x3, p3 = self.encode3(p2)

        x = self.middle(p3)

        d3 = self.decode3(x, x3)
        d2 = self.decode2(d3, x2)
        d1 = self.decode1(d2, x1)
        d0 = self.decode0(d1, x0)
        return d0


class LeapUnet1D(nn.Module):
    def __init__(self, in_chans: tuple[int, ...] = (25, 32, 64, 128, 256, 512)) -> None:
        super().__init__()

        self.unet1d = Unet1D(in_chans, kernel_size=3, stride=1)
        self.cnn1 = nn.Conv1d(in_chans[1], 14, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=True, num_layers=2)
        self.fc = nn.Linear(64 * 2, 60)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor, random_idx: npt.NDArray[np.integer] | None = None) -> torch.Tensor:
        """
        Args:
            x: (bs, 556)

        Returns:
            x: (bs, 368)

        -- (N, 32, 64) ------------------------------------------------------> (N, 32, 64) -> [cnn] -> (N, 14, 64) -> [Head] -> (N, 368)
            |                                                                       |
            --> (N, 64, 32) --------------------------------------------> (N, 64, 32)
                |                                                                |
                --> (N, 128, 16) -----------------------------------> (N, 128, 16)
                    |                                                      |
                    --> (N, 256, 4) --------------------------> (N, 256, 8)
                        |                                             |
                        ------> [Middle] ---> (N, 512, 4) ------------
        """

        # (N, 25, 60)
        x = x_to_seq(x).permute(0, 2, 1)
        x = nn.functional.pad(x, (0, 4), "constant", 0)

        # (N, 32, 64)
        x = self.unet1d(x)
        # (N, 32, 128)
        x, _ = self.lstm(x)
        # (N, 14, 128)
        pall = self.cnn1(x)
        # (N, 14, 60)
        pall = self.fc(pall)
        # (n, 360)
        p_seq = self.flatten(pall[:, :6, :])
        # (n, 8)
        p_flat = pall[:, 6:14, :].mean(dim=-1)
        out = torch.cat([p_seq, p_flat], dim=-1)
        return out


if __name__ == "__main__":
    from torchinfo import summary

    data = torch.randn(32, 556)
    x = x_to_seq(data).permute(0, 2, 1)
    x = nn.functional.pad(x, (0, 4), "constant", 0)

    index_grps = np.array_split(np.arange(60), 3)
    random_idx = np.array([np.random.choice(idx_grp) for idx_grp in index_grps])

    in_chans = (25, 64, 128, 256, 512, 1024)
    model = LeapUnet1D(in_chans).cpu()
    print(model)
    summary(model, (8, 556), device="cpu")
    out = model(data, random_idx)
    print(f"{out.shape = }")
    assert out.shape == (32, 368)
