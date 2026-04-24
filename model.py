"""
model.py
========
Self-Pruning Neural Network — Tredence Analytics Case Study
Author: Senior AI Research Engineer

Architecture:
  CNN Backbone (Conv + BN + ReLU + Pooling)
      └─> Multiple PrunableLinear layers in the classifier head

The PrunableLinear layer learns gate_scores alongside weights.
During forward pass, sigmoid-gated weights are used, allowing the
network to prune itself by driving gate values toward zero.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  PrunableLinear Layer
# ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that learns to prune itself.

    Each weight w_ij is multiplied by a learnable gate g_ij ∈ (0, 1).
    The gate is computed as sigmoid(gate_score_ij). When sparsity
    regularisation drives gate_scores to -∞, the effective weight
    contribution vanishes — the connection is "pruned".

    Gradients flow through both `weight` and `gate_scores` via the
    standard autograd chain, requiring no custom backward pass.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard linear parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features)) if bias else None

        # Learnable gate scores — same shape as weight
        # Initialised near 0 so sigmoid ≈ 0.5 (open gates at start)
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=0.01)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        # Slightly positive init → gates start ~0.55, gently open
        nn.init.constant_(self.gate_scores, 0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Convert raw scores to gate values in (0, 1)
        gates = torch.sigmoid(self.gate_scores)          # shape: (out, in)

        # Step 2: Apply gates element-wise to the weight matrix
        pruned_weights = self.weight * gates             # shape: (out, in)

        # Step 3: Standard linear operation with pruned weights
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return the current gate values (detached, for analysis)."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below `threshold` (pruned connections)."""
        gates = self.get_gates()
        return (gates < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bias={self.bias is not None}")


# ─────────────────────────────────────────────
#  CNN Backbone
# ─────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv → BN → ReLU (optionally with residual shortcut)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class SelfPruningNet(nn.Module):
    """
    CNN backbone followed by a deep PrunableLinear classifier.

    Structure
    ---------
    Backbone  : 3 ResNet-style blocks (64 → 128 → 256 channels)
    Classifier: Conv → Pool → Flatten → PrunableLinear × 3

    The three PrunableLinear layers are where sparsity is learned.
    Their combined parameter count is substantial, making pruning
    meaningful rather than cosmetic.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()

        # ── Stem ──────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ── Residual blocks ───────────────────────────────────
        self.layer1 = self._make_layer(64,  64,  num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)

        # ── Transition conv + global average pool ─────────────
        self.transition = nn.Sequential(
            nn.Conv2d(256, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),          # → (B, 512, 4, 4)
        )

        flat_dim = 512 * 4 * 4                 # 8 192

        # ── Prunable classifier head ───────────────────────────
        # Three PrunableLinear layers — the sparsity lives here
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            PrunableLinear(flat_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            PrunableLinear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            PrunableLinear(512, num_classes),
        )

        self._init_weights()

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, num_blocks: int, stride: int):
        layers = [ConvBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(ConvBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.transition(x)
        x = x.view(x.size(0), -1)   # flatten
        x = self.classifier(x)
        return x

    # ── Introspection helpers ──────────────────────────────────

    def prunable_layers(self) -> list[PrunableLinear]:
        """Return all PrunableLinear modules in the network."""
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of ALL gate values across every PrunableLinear layer.
        Since sigmoid(x) > 0 always, L1 = sum = mean * numel.
        Minimising this loss drives gates toward 0, pruning the network.
        """
        total = torch.tensor(0.0, requires_grad=True)
        for layer in self.prunable_layers():
            gates = torch.sigmoid(layer.gate_scores)
            total = total + gates.sum()
        return total

    def global_sparsity(self, threshold: float = 1e-2) -> float:
        """Overall fraction of pruned (gate < threshold) connections."""
        pruned = total = 0
        for layer in self.prunable_layers():
            g = layer.get_gates()
            pruned += (g < threshold).sum().item()
            total  += g.numel()
        return pruned / total if total > 0 else 0.0

    def gate_histogram_data(self) -> torch.Tensor:
        """Concatenate all gate values for histogram analysis."""
        return torch.cat(
            [torch.sigmoid(l.gate_scores).detach().cpu().flatten()
             for l in self.prunable_layers()]
        )
