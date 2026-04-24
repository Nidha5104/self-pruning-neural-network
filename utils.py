"""
utils.py
========
Utility functions for the Self-Pruning Neural Network project.

Covers:
  - Sparsity computation (per-layer and global)
  - Training metrics bookkeeping
  - Gate-value histogram generation
  - Early-stopping helper
  - Result table pretty-printing
"""

from __future__ import annotations

import math
from typing import Optional

import matplotlib
matplotlib.use("Agg")           # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────
#  Sparsity Utilities
# ─────────────────────────────────────────────────────────────────

def compute_sparsity(model: nn.Module, threshold: float = 1e-2) -> dict:
    """
    Compute sparsity statistics for every PrunableLinear layer and globally.

    Returns
    -------
    dict with keys:
        'global'   : overall sparsity fraction (float)
        'layers'   : list of per-layer sparsity fractions
        'n_pruned' : total number of pruned connections
        'n_total'  : total number of prunable connections
    """
    from model import PrunableLinear

    pruned_total = 0
    param_total  = 0
    layer_stats  = []

    for name, module in model.named_modules():
        if isinstance(module, PrunableLinear):
            gates   = torch.sigmoid(module.gate_scores).detach()
            pruned  = (gates < threshold).sum().item()
            total   = gates.numel()
            frac    = pruned / total

            layer_stats.append({
                "name"    : name,
                "pruned"  : pruned,
                "total"   : total,
                "sparsity": frac,
            })

            pruned_total += pruned
            param_total  += total

    global_sparsity = pruned_total / param_total if param_total > 0 else 0.0

    return {
        "global"   : global_sparsity,
        "layers"   : layer_stats,
        "n_pruned" : pruned_total,
        "n_total"  : param_total,
    }


def get_all_gate_values(model: nn.Module) -> torch.Tensor:
    """Return a flat tensor of all gate values from every PrunableLinear."""
    from model import PrunableLinear
    all_gates = []
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            all_gates.append(
                torch.sigmoid(module.gate_scores).detach().cpu().flatten()
            )
    return torch.cat(all_gates) if all_gates else torch.tensor([])


# ─────────────────────────────────────────────────────────────────
#  Metrics & Bookkeeping
# ─────────────────────────────────────────────────────────────────

class MetricsTracker:
    """
    Lightweight container for epoch-level metrics.
    Stores train/val loss, accuracy, and sparsity per epoch.
    """

    def __init__(self):
        self.train_loss     : list[float] = []
        self.train_acc      : list[float] = []
        self.val_loss       : list[float] = []
        self.val_acc        : list[float] = []
        self.sparsity       : list[float] = []
        self.sparsity_loss  : list[float] = []

    def update(self, *, train_loss: float, train_acc: float,
               val_loss: float, val_acc: float,
               sparsity: float, sp_loss: float):
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.sparsity.append(sparsity)
        self.sparsity_loss.append(sp_loss)

    def best_val_acc(self) -> float:
        return max(self.val_acc) if self.val_acc else 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy for a batch."""
    preds = outputs.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ─────────────────────────────────────────────────────────────────
#  Early Stopping
# ─────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Monitors a metric (higher = better by default) and signals
    when training should stop due to lack of improvement.

    Parameters
    ----------
    patience : int   — epochs to wait without improvement
    min_delta: float — minimum change to count as improvement
    mode     : str   — 'max' (accuracy) or 'min' (loss)
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4,
                 mode: str = "max"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best       = -math.inf if mode == "max" else math.inf
        self.counter    = 0
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        improved = (
            metric > self.best + self.min_delta if self.mode == "max"
            else metric < self.best - self.min_delta
        )
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ─────────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────────

PALETTE = {
    "bg"        : "#0f0f14",
    "panel"     : "#1a1a24",
    "accent1"   : "#ff6b35",
    "accent2"   : "#4ecdc4",
    "accent3"   : "#ffe66d",
    "text"      : "#e8e8f0",
    "grid"      : "#2a2a3a",
}

plt.rcParams.update({
    "figure.facecolor"  : PALETTE["bg"],
    "axes.facecolor"    : PALETTE["panel"],
    "axes.edgecolor"    : PALETTE["grid"],
    "axes.labelcolor"   : PALETTE["text"],
    "xtick.color"       : PALETTE["text"],
    "ytick.color"       : PALETTE["text"],
    "text.color"        : PALETTE["text"],
    "grid.color"        : PALETTE["grid"],
    "grid.linestyle"    : "--",
    "grid.alpha"        : 0.5,
    "font.family"       : "monospace",
    "axes.titlesize"    : 13,
    "axes.labelsize"    : 11,
})


def plot_training_curves(metrics: MetricsTracker,
                         lam: float,
                         save_path: str = "training_curves.png") -> None:
    """
    Four-panel training dashboard:
      Top-left  : Train/Val loss
      Top-right : Train/Val accuracy
      Bot-left  : Sparsity % over epochs
      Bot-right : Sparsity loss component
    """
    epochs = list(range(1, len(metrics.train_loss) + 1))

    fig = plt.figure(figsize=(14, 8), facecolor=PALETTE["bg"])
    fig.suptitle(f"Training Dashboard  |  λ = {lam}",
                 color=PALETTE["text"], fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    def _ax(pos, title, ylabel, xlabel="Epoch"):
        ax = fig.add_subplot(pos)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        return ax

    # Loss
    ax0 = _ax(gs[0, 0], "Loss", "Loss")
    ax0.plot(epochs, metrics.train_loss, color=PALETTE["accent1"],
             lw=2, label="Train")
    ax0.plot(epochs, metrics.val_loss,   color=PALETTE["accent2"],
             lw=2, linestyle="--", label="Val")
    ax0.legend(framealpha=0.2)

    # Accuracy
    ax1 = _ax(gs[0, 1], "Accuracy", "Accuracy (%)")
    ax1.plot(epochs,
             [a * 100 for a in metrics.train_acc],
             color=PALETTE["accent1"], lw=2, label="Train")
    ax1.plot(epochs,
             [a * 100 for a in metrics.val_acc],
             color=PALETTE["accent2"], lw=2, linestyle="--", label="Val")
    ax1.legend(framealpha=0.2)

    # Sparsity
    ax2 = _ax(gs[1, 0], "Global Sparsity", "Sparsity (%)")
    ax2.plot(epochs,
             [s * 100 for s in metrics.sparsity],
             color=PALETTE["accent3"], lw=2)
    ax2.fill_between(epochs,
                     [s * 100 for s in metrics.sparsity],
                     alpha=0.15, color=PALETTE["accent3"])

    # Sparsity loss
    ax3 = _ax(gs[1, 1], "Sparsity Loss Component", "λ · SparsityLoss")
    ax3.plot(epochs, metrics.sparsity_loss,
             color="#c77dff", lw=2)

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    print(f"[plot] Training curves saved → {save_path}")


def plot_gate_histogram(model: nn.Module,
                        lam: float,
                        save_path: str = "gate_histogram.png") -> None:
    """
    Histogram of ALL gate values in the trained model.

    A successful result shows a large spike near 0 (pruned weights)
    and a smaller cluster near 1 (important retained connections).
    """
    gate_vals = get_all_gate_values(model).numpy()

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])

    n_bins   = 80
    counts, edges = np.histogram(gate_vals, bins=n_bins, range=(0, 1))
    bin_centers   = 0.5 * (edges[:-1] + edges[1:])

    # Color bars by region: pruned (near 0) vs retained (near 1)
    colors = [PALETTE["accent1"] if c < 0.05 else
              PALETTE["accent2"] if c > 0.80 else
              PALETTE["accent3"]
              for c in bin_centers]

    ax.bar(bin_centers, counts, width=(edges[1] - edges[0]) * 0.9,
           color=colors, alpha=0.85, edgecolor=PALETTE["bg"], linewidth=0.3)

    ax.set_title(f"Gate Value Distribution  |  λ = {lam}",
                 color=PALETTE["text"], fontsize=14)
    ax.set_xlabel("Gate Value g = σ(score)")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y")

    # Annotation
    n_pruned = (gate_vals < 0.01).sum()
    pct      = 100 * n_pruned / len(gate_vals)
    ax.axvline(0.01, color="#ff4444", linestyle="--", lw=1.5,
               label=f"Threshold 0.01  ({pct:.1f}% pruned)")
    ax.legend(framealpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    print(f"[plot] Gate histogram saved → {save_path}")


def plot_lambda_tradeoff(results: list[dict],
                         save_path: str = "lambda_tradeoff.png") -> None:
    """
    Scatter + line plot showing Accuracy vs Sparsity for each λ.
    results: list of {lambda, accuracy, sparsity}
    """
    lambdas    = [r["lambda"]   for r in results]
    accuracies = [r["accuracy"] for r in results]
    sparsities = [r["sparsity"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=PALETTE["bg"])
    fig.suptitle("λ Trade-off: Accuracy vs Sparsity",
                 color=PALETTE["text"], fontsize=14, fontweight="bold")

    for ax in axes:
        ax.set_facecolor(PALETTE["panel"])
        ax.grid(True)

    # Accuracy vs Lambda
    ax = axes[0]
    ax.plot(lambdas, [a * 100 for a in accuracies],
            color=PALETTE["accent1"], lw=2, marker="o", markersize=8)
    for lam, acc in zip(lambdas, accuracies):
        ax.annotate(f"{acc*100:.1f}%",
                    xy=(lam, acc * 100), xytext=(4, 4),
                    textcoords="offset points",
                    color=PALETTE["text"], fontsize=9)
    ax.set_xlabel("λ (lambda)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy vs λ")

    # Sparsity vs Lambda
    ax = axes[1]
    ax.plot(lambdas, [s * 100 for s in sparsities],
            color=PALETTE["accent2"], lw=2, marker="s", markersize=8)
    for lam, sp in zip(lambdas, sparsities):
        ax.annotate(f"{sp*100:.1f}%",
                    xy=(lam, sp * 100), xytext=(4, 4),
                    textcoords="offset points",
                    color=PALETTE["text"], fontsize=9)
    ax.set_xlabel("λ (lambda)")
    ax.set_ylabel("Sparsity (%)")
    ax.set_title("Sparsity vs λ")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    print(f"[plot] Lambda trade-off plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────
#  Result Table
# ─────────────────────────────────────────────────────────────────

def print_results_table(results: list[dict]) -> None:
    """Pretty-print the λ / Accuracy / Sparsity comparison table."""
    header = f"{'Lambda':>10}  {'Test Acc (%)':>14}  {'Sparsity (%)':>14}"
    sep    = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        print(f"{r['lambda']:>10.4f}  {r['accuracy']*100:>14.2f}  "
              f"{r['sparsity']*100:>14.2f}")
    print(sep)
