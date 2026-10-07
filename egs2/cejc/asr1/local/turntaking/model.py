#!/usr/bin/env python
"""VAP-lite turn-taking head（凍結 ASR エンコーダの上に載せる学習対象ブランチ）.

入力: 凍結 encoder_out (B, T, 256)  ※frame=33ms, causal, look_ahead≈300ms
出力:
  cls_logits (B, T, 3)        turn-end / 相槌 / 継続 を毎フレーム
  va_logits  (B, T, n_bins)   対象話者が「今後 h 内に発話するか」(自己教師)

CIF は使わない（沈黙で発火しないため）。frame-level に切り替えるのが本設計の肝。
時間方向は因果畳み込みのみ（未来を見ない＝ストリーミング整合）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, ch, k, dilation):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(ch, ch, k, dilation=dilation)

    def forward(self, x):                 # x: (B, C, T)
        x = F.pad(x, (self.pad, 0))       # 左パディングのみ = 因果
        return self.conv(x)


class TurnTakingHead(nn.Module):
    def __init__(self, d_in=256, d_hidden=128, n_classes=3, va_bins=4,
                 n_layers=3, kernel=5, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_hidden)
        self.convs = nn.ModuleList(
            [CausalConv1d(d_hidden, kernel, dilation=2 ** i) for i in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_hidden) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.cls_head = nn.Linear(d_hidden, n_classes)
        self.va_head = nn.Linear(d_hidden, va_bins)

    def forward(self, x):                 # x: (B, T, d_in)
        h = self.in_proj(x)               # (B, T, H)
        y = h.transpose(1, 2)             # (B, H, T)
        for conv, norm in zip(self.convs, self.norms):
            z = F.relu(conv(y))
            y = y + self.drop(z)          # residual
            y = norm(y.transpose(1, 2)).transpose(1, 2)
        h = y.transpose(1, 2)             # (B, T, H)
        return self.cls_head(h), self.va_head(h)


if __name__ == "__main__":              # 形状スモークテスト
    m = TurnTakingHead()
    x = torch.randn(2, 37, 256)
    c, v = m(x)
    n = sum(p.numel() for p in m.parameters())
    print(f"cls={tuple(c.shape)} va={tuple(v.shape)} params={n/1e6:.3f}M")
