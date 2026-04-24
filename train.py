"""
train.py
========
Self-Pruning Neural Network — Training & Evaluation Pipeline
Tredence Analytics Case Study

Usage
-----
# Single run (default λ = 1e-4):
    python train.py

# Sweep over three λ values:
    python train.py --sweep

# Custom λ + epochs:
    python train.py --lam 5e-4 --epochs 30

Outputs (per run)
-----------------
  checkpoints/best_model_lam{λ}.pt  — best checkpoint
  training_curves_lam{λ}.png        — loss/acc/sparsity curves
  gate_histogram_lam{λ}.png         — gate distribution plot
  lambda_tradeoff.png               — accuracy vs sparsity across λ (sweep only)
"""

import argparse
import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from model import SelfPruningNet
from utils import (
    MetricsTracker,
    EarlyStopping,
    accuracy,
    compute_sparsity,
    plot_training_curves,
    plot_gate_histogram,
    plot_lambda_tradeoff,
    print_results_table,
)

# ─────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    lam        = 1e-4,    # sparsity regularisation weight
    epochs     = 20,
    batch_size = 128,
    lr         = 3e-4,
    weight_decay = 1e-4,
    val_frac   = 0.1,     # fraction of train set used for validation
    threshold  = 1e-2,    # gate < threshold → pruned
    num_workers = 4,
    dropout    = 0.3,
    patience   = 8,       # early-stopping patience
    seed       = 42,
)

SWEEP_LAMBDAS = [1e-5, 1e-4, 5e-4]   # low / medium / high


# ─────────────────────────────────────────────────────────────────
#  Data
# ─────────────────────────────────────────────────────────────────

def get_dataloaders(batch_size: int, val_frac: float, num_workers: int):
    """
    CIFAR-10 with aggressive augmentation for the training split:
      - RandomCrop + horizontal flip (standard)
      - RandAugment (stronger, helps generalisation)
      - Normalise to ImageNet stats (widely used for CIFAR with CNNs)
    """
    MEAN = (0.4914, 0.4822, 0.4465)
    STD  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    full_train = datasets.CIFAR10(root="./data", train=True,
                                  download=True, transform=train_tf)
    test_set   = datasets.CIFAR10(root="./data", train=False,
                                  download=True, transform=test_tf)

    n_val  = int(len(full_train) * val_frac)
    n_tr   = len(full_train) - n_val
    train_set, val_set = random_split(full_train, [n_tr, n_val],
                                      generator=torch.Generator().manual_seed(42))

    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=True, persistent_workers=(num_workers > 0))

    train_loader = DataLoader(train_set, shuffle=True,  **kw)
    val_loader   = DataLoader(val_set,   shuffle=False, **kw)
    test_loader  = DataLoader(test_set,  shuffle=False, **kw)

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────
#  Training / Evaluation Loops
# ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer,
                    scheduler, lam, device, scaler):
    """
    One epoch of training with:
      - Mixed-precision (AMP) via GradScaler
      - Total Loss = CrossEntropy + λ * L1(gates)
    """
    model.train()
    total_loss = sp_loss_sum = correct = seen = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), \
                       labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            logits    = model(imgs)
            ce_loss   = criterion(logits, labels)
            sp_loss   = model.sparsity_loss()
            loss      = ce_loss + lam * sp_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs           = labels.size(0)
        total_loss  += loss.item() * bs
        sp_loss_sum += (lam * sp_loss).item() * bs
        correct     += (logits.argmax(1) == labels).sum().item()
        seen        += bs

    return total_loss / seen, correct / seen, sp_loss_sum / seen


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on val / test set (no gradient tracking)."""
    model.eval()
    total_loss = correct = seen = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), \
                       labels.to(device, non_blocking=True)
        logits     = model(imgs)
        loss       = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        seen       += labels.size(0)

    return total_loss / seen, correct / seen


# ─────────────────────────────────────────────────────────────────
#  Main Training Function
# ─────────────────────────────────────────────────────────────────

def run_experiment(lam: float, cfg: dict, device: torch.device) -> dict:
    """
    Full training + evaluation for a single λ value.

    Returns
    -------
    dict: {lambda, accuracy, sparsity, metrics_tracker}
    """
    print(f"\n{'═'*60}")
    print(f"  Experiment:  λ = {lam:.2e}  |  {cfg['epochs']} epochs")
    print(f"{'═'*60}")

    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])

    # ── Data ─────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg["batch_size"], cfg["val_frac"], cfg["num_workers"]
    )

    # ── Model ────────────────────────────────────────────────────
    model = SelfPruningNet(num_classes=10, dropout=cfg["dropout"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params : {n_params:,}")

    # ── Optimiser + Scheduler ─────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])

    # OneCycleLR gives a fast warm-up and cosine annealing in one pass
    total_steps = cfg["epochs"] * len(train_loader)
    scheduler   = OneCycleLR(optimizer, max_lr=cfg["lr"] * 10,
                             total_steps=total_steps,
                             pct_start=0.1,
                             anneal_strategy="cos")
    scaler      = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Tracking ──────────────────────────────────────────────────
    metrics      = MetricsTracker()
    early_stop   = EarlyStopping(patience=cfg["patience"], mode="max")
    best_val_acc = 0.0
    best_state   = None

    Path("checkpoints").mkdir(exist_ok=True)
    ckpt_path = f"checkpoints/best_model_lam{lam:.2e}.pt"

    # ── Training loop ─────────────────────────────────────────────
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        tr_loss, tr_acc, sp_loss_val = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scheduler, lam, device, scaler
        )
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)

        sparsity = model.global_sparsity(threshold=cfg["threshold"])

        metrics.update(
            train_loss=tr_loss, train_acc=tr_acc,
            val_loss=vl_loss,   val_acc=vl_acc,
            sparsity=sparsity,  sp_loss=sp_loss_val,
        )

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(
            f"  Epoch {epoch:>3}/{cfg['epochs']}  "
            f"| TrLoss {tr_loss:.4f}  TrAcc {tr_acc*100:5.2f}%  "
            f"| VlLoss {vl_loss:.4f}  VlAcc {vl_acc*100:5.2f}%  "
            f"| Sparsity {sparsity*100:5.1f}%  "
            f"| LR {lr_now:.2e}  "
            f"| {elapsed:.1f}s"
        )

        # Save best checkpoint
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)

        # Early stopping check
        if early_stop(vl_acc):
            print(f"  [early stop] No improvement for {cfg['patience']} epochs.")
            break

    # ── Final evaluation on test set ──────────────────────────────
    print(f"\n  Loading best checkpoint ({ckpt_path}) …")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    _, test_acc = evaluate(model, test_loader, criterion, device)
    sparsity    = compute_sparsity(model, threshold=cfg["threshold"])

    print(f"\n  ── Final Results ──────────────────────────────────")
    print(f"  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Global Sparsity: {sparsity['global']*100:.2f}%  "
          f"({sparsity['n_pruned']:,} / {sparsity['n_total']:,} connections pruned)")

    for ls in sparsity["layers"]:
        print(f"    {ls['name']:<45}  {ls['sparsity']*100:5.1f}%")

    # ── Plots ─────────────────────────────────────────────────────
    lam_tag = f"lam{lam:.2e}"
    plot_training_curves(metrics, lam, save_path=f"training_curves_{lam_tag}.png")
    plot_gate_histogram(model,    lam, save_path=f"gate_histogram_{lam_tag}.png")

    return {
        "lambda"   : lam,
        "accuracy" : test_acc,
        "sparsity" : sparsity["global"],
        "metrics"  : metrics,
    }


# ─────────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Self-Pruning Neural Network — Tredence Analytics"
    )
    parser.add_argument("--lam",        type=float, default=DEFAULTS["lam"])
    parser.add_argument("--epochs",     type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--batch_size", type=int,   default=DEFAULTS["batch_size"])
    parser.add_argument("--lr",         type=float, default=DEFAULTS["lr"])
    parser.add_argument("--dropout",    type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--threshold",  type=float, default=DEFAULTS["threshold"])
    parser.add_argument("--patience",   type=int,   default=DEFAULTS["patience"])
    parser.add_argument("--sweep",      action="store_true",
                        help="Run experiments for all λ values in SWEEP_LAMBDAS")
    parser.add_argument("--no_cuda",    action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu"
    )
    print(f"\n  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    cfg = {
        "epochs"      : args.epochs,
        "batch_size"  : args.batch_size,
        "lr"          : args.lr,
        "weight_decay": DEFAULTS["weight_decay"],
        "val_frac"    : DEFAULTS["val_frac"],
        "threshold"   : args.threshold,
        "num_workers" : DEFAULTS["num_workers"],
        "dropout"     : args.dropout,
        "patience"    : args.patience,
        "seed"        : DEFAULTS["seed"],
    }

    lambdas = SWEEP_LAMBDAS if args.sweep else [args.lam]
    results = []

    for lam in lambdas:
        result = run_experiment(lam, cfg, device)
        results.append({
            "lambda"   : result["lambda"],
            "accuracy" : result["accuracy"],
            "sparsity" : result["sparsity"],
        })

    # Print summary table
    print_results_table(results)

    # Save results to JSON
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Results saved → results.json")

    # Trade-off plot (meaningful only for sweep)
    if len(results) > 1:
        plot_lambda_tradeoff(results, save_path="lambda_tradeoff.png")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
