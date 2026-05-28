import torch
import torch.nn as nn
import torch.nn.functional as functional

class LinearAttnPool1D(nn.Module):
    def __init__(
        self, 
        channels, 
        hidden=16, 
        dropout=0.2,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(channels)
        self.q = nn.Sequential(nn.Linear(channels, hidden), nn.Dropout(p=dropout))
        self.k = nn.Sequential(nn.Linear(channels, hidden), nn.Dropout(p=dropout))
        self.v = nn.Sequential(nn.Linear(channels, channels), nn.Dropout(p=dropout))
        self.w = nn.Linear(channels, channels)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def phi(self, x):
        # return functional.elu(x) + 1
        return functional.softplus(x)

    def forward(self, x):
        # [B, hidden_channels, n_mels]
        x = self.ln(x)

        Q = self.phi(self.q(x))  # [B, L, H]
        K = self.phi(self.k(x))  # [B, L, H]
        V = self.v(x)            # [B, L, C]

        KV = torch.einsum('blh,blc->hc', K, V)      # [H, C]
        out = torch.einsum('blh,hc->blc', Q, KV)    # [B, L, C]

        weights = torch.softmax(self.w(out), dim=1) # [B, L, 1]
        out = (out * weights).sum(dim=1)
        
        return out

class ResidualBlock1D(nn.Module):
    def __init__(self, c, k, dropout=0.1):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(c)
        self.conv1 = nn.Conv1d(c, c, kernel_size=k, padding=k//2)
        self.bn2 = nn.BatchNorm1d(c)
        self.conv2 = nn.Conv1d(c, c, kernel_size=k, padding=k//2)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = functional.gelu(out) # Gaussian Error Linear Unit
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = functional.gelu(out)
        out = self.dropout(out)

        # Stabilize residual connection
        out = residual + self.alpha * out
        out = functional.gelu(out)
        return out
    
class WaveformResNet(nn.Module):
    def __init__(
        self, 
        bin_shape=64,
        mel_shape=128,
        hidden_channels=16,
        conv_kernel=3,
        n_resBlocks=3,
        gru_hidden=16,
        gru_layers=3,
        attn_hidden=16,
        dropout=0.1
    ):
        super().__init__()
        self.bin_shape = bin_shape
        self.mel_shape = mel_shape

        self.conv1x1 = nn.Conv1d(
            bin_shape, hidden_channels, conv_kernel, padding=conv_kernel//2
        )
        self.res_blocks = nn.ModuleList(
            [ResidualBlock1D(hidden_channels, conv_kernel, dropout) for _ in range(n_resBlocks)]
        )
        self.gru = nn.GRU(
            input_size=mel_shape, 
            hidden_size=hidden_channels, 
            num_layers=gru_layers, 
            batch_first=True,
            bidirectional=False
        )
        self.attn_pool = LinearAttnPool1D(
            channels=gru_hidden, 
            hidden=attn_hidden, 
            dropout=dropout
        )
        self.out = nn.Linear(gru_hidden, 1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for block in self.res_blocks:
            block.init_weights()

        self.attn_pool.init_weights()

    def forward(self, x):
        # x: [B, n_mels, T]

        # Padding
        B, C, T = x.shape
        if (T < self.bin_shape):
            pad = torch.zeros(B, C, self.bin_shape - T, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=2)  # dim=2 is the time axis
            # x = functional.pad(x, (0, self.bin_shape - T))
        elif (T > self.bin_shape):
            x = x[:, :, :self.bin_shape]
        
        # Feature extraction with residual blocks
        feat = x.permute(0, 2, 1)       # [B, T, n_mels]
        feat = self.conv1x1(feat)       # [B, C_T, n_mels]

        # Learn spec features
        for block in self.res_blocks:
            feat = block(feat)         # [B, C_T, n_mels]

        # Learn dynamics of spec features
        gru_out, _ = self.gru(feat)    # [B, C_T, gru_hidden]

        # Pool features
        feat = self.attn_pool(gru_out) # [B, attn_hidden]

        # Head
        logits = self.out(feat) # [B, 1]
        return logits
