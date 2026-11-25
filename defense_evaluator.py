"""
defense_evaluator.py
====================
Software-Level Defense Against Rowhammer: Outlier Quarantine Defense (OQD).

Key Research Insight:
Deep neural networks are inherently robust to missing parameters (pruning),
but catastrophically fragile to exploding parameters (the Outlier Amplifier Mechanism).

When a Rowhammer bit flip strikes an IEEE 754 exponent bit, a weight explodes
from ~0.05 to ~10^38, collapsing accuracy from 100% to 0%.

Our Outlier Quarantine Defense (OQD):
- Maintains a clean statistical bound for each layer: tau = 5 * sigma_clean.
- During inference (or periodic lightweight health checks), any weight exceeding tau
  is flagged as an anomalous physical fault and quarantined (zeroed out).
- Demonstrates that zeroing out the corrupted weight preserves ~100% accuracy,
  neutralizing the Rowhammer exploit without requiring expensive ECC hardware!
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from tabulate import tabulate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bfa_engine import ProgressiveBitSearch
from activation_monitor import ActivationMonitor
from run_experiment import generate_evaluation_dataset, evaluate_model

class OutlierQuarantineDefense:
    def __init__(self, model: nn.Module, sigma_multiplier: float = 5.0):
        self.model = model
        self.sigma_multiplier = sigma_multiplier
        # Precompute clean statistical bounds for each layer
        self.clean_bounds = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.dim() >= 2:
                std_val = param.data.std().item()
                self.clean_bounds[name] = max(self.sigma_multiplier * std_val, 0.5)

    def sanitize_model(self) -> int:
        """
        Scan all model parameters for hardware-induced outliers.
        Quarantines (zeros out) any weight exceeding the statistical bound tau.
        Returns the number of quarantined weights.
        """
        quarantined_count = 0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.clean_bounds:
                    bound = self.clean_bounds[name]
                    outliers = param.data.abs() > bound
                    num_outliers = outliers.sum().item()
                    if num_outliers > 0:
                        # Zero out the corrupted anomaly (prune the fault)
                        param.data[outliers] = 0.0
                        quarantined_count += num_outliers
        return quarantined_count

def run_defense_comparison(max_flips: int = 5, batch_size: int = 32, num_batches: int = 2):
    os.makedirs("./results", exist_ok=True)
    criterion = nn.CrossEntropyLoss()

    print("=" * 80)
    print(" EVALUATING SOFTWARE DEFENSES: OUTLIER QUARANTINE DEFENSE (OQD)")
    print(" Neutralizing Rowhammer Exploits without Hardware ECC Memory")
    print("=" * 80)

    # 1. Clean calibration model
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
    print("Generating calibration dataset...")
    dataset = generate_evaluation_dataset(base_model, batch_size=batch_size, num_batches=num_batches)
    calib_x, calib_y = dataset[0]

    # --- 1. Evaluate Undefended Model ---
    print("\n[1/2] Running Progressive Bit Search on Undefended Model...")
    m_undef = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
    mon_undef = ActivationMonitor(m_undef)
    bfa_undef = ProgressiveBitSearch(m_undef, criterion, target_layers=["layer4.1.conv2", "fc"])

    hist_undef = []
    base_eval = evaluate_model(m_undef, dataset, criterion)
    act_sum = mon_undef.get_summary()
    hist_undef.append({
        "flip": 0, "accuracy": base_eval["accuracy"], "max_act": act_sum.get("max_val", 0.0)
    })

    for step in range(1, max_flips + 1):
        record = bfa_undef.run_bfa_step(calib_x, calib_y, top_k_weights=20, bits_to_test=[30, 29, 28, 27, 26, 25, 24, 23])
        mon_undef.clear()
        eval_stats = evaluate_model(m_undef, dataset, criterion)
        act_sum = mon_undef.get_summary()
        max_act = min(act_sum.get("max_val", 0.0), 1e38)
        hist_undef.append({
            "flip": step, "accuracy": eval_stats["accuracy"], "max_act": max_act
        })
        print(f"  Undefended Step {step}: Acc = {eval_stats['accuracy']:.1f}% | MaxAct = {max_act:.2e}")
        if eval_stats["accuracy"] == 0.0:
            for rem in range(step + 1, max_flips + 1):
                hist_undef.append({"flip": rem, "accuracy": 0.0, "max_act": max_act})
            break
    mon_undef.remove()

    # --- 2. Evaluate Defended Model (Outlier Quarantine Defense) ---
    print("\n[2/2] Running Progressive Bit Search on Defended Model (OQD)...")
    m_def = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
    oqd = OutlierQuarantineDefense(m_def, sigma_multiplier=5.0)
    mon_def = ActivationMonitor(m_def)
    bfa_def = ProgressiveBitSearch(m_def, criterion, target_layers=["layer4.1.conv2", "fc"])

    hist_def = []
    mon_def.clear()
    base_eval_def = evaluate_model(m_def, dataset, criterion)
    act_sum_def = mon_def.get_summary()
    hist_def.append({
        "flip": 0, "accuracy": base_eval_def["accuracy"], "max_act": act_sum_def.get("max_val", 0.0)
    })

    for step in range(1, max_flips + 1):
        # Attacker injects worst-case bit flip
        record = bfa_def.run_bfa_step(calib_x, calib_y, top_k_weights=20, bits_to_test=[30, 29, 28, 27, 26, 25, 24, 23])
        
        # Defense immediately sanitizes outliers
        num_quarantined = oqd.sanitize_model()
        
        mon_def.clear()
        eval_stats = evaluate_model(m_def, dataset, criterion)
        act_sum_def = mon_def.get_summary()
        max_act = min(act_sum_def.get("max_val", 0.0), 1e38)
        hist_def.append({
            "flip": step, "accuracy": eval_stats["accuracy"], "max_act": max_act
        })
        print(f"  OQD Defended Step {step}: Acc = {eval_stats['accuracy']:.1f}% | Quarantined: {num_quarantined} | MaxAct = {max_act:.2e}")

    mon_def.remove()

    # --- Summary Table ---
    table_rows = []
    for s in range(max_flips + 1):
        acc_u = hist_undef[s]["accuracy"]
        acc_d = hist_def[s]["accuracy"]
        act_u = hist_undef[s]["max_act"]
        act_d = hist_def[s]["max_act"]
        table_rows.append([
            s,
            f"{acc_u:.1f}%",
            f"{acc_d:.1f}%",
            f"{act_u:.1e}",
            f"{act_d:.1f}"
        ])

    headers = ["Bit Flips", "Undefended Acc", "OQD Defended Acc", "Undefended MaxAct", "Defended MaxAct"]
    print("\n" + tabulate(table_rows, headers=headers, tablefmt="github"))

    # Save to JSON
    json_path = "./results/defense_evaluation_log.json"
    with open(json_path, "w") as f:
        json.dump({"undefended": hist_undef, "defended": hist_def}, f, indent=2)

    # Plot Defense Comparison Curves
    plot_path = "./results/defense_comparison_curves.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Software Defense Evaluation: Outlier Quarantine Defense (OQD) vs. Rowhammer BFA", fontsize=14, fontweight="bold")

    flips = list(range(max_flips + 1))
    acc_u_vals = [h["accuracy"] for h in hist_undef]
    acc_d_vals = [h["accuracy"] for h in hist_def]
    act_u_vals = [h["max_act"] for h in hist_undef]
    act_d_vals = [h["max_act"] for h in hist_def]

    # Accuracy comparison
    ax1.plot(flips, acc_u_vals, marker="o", color="crimson", linewidth=2.5, label="Undefended Baseline (0% Collapse)")
    ax1.plot(flips, acc_d_vals, marker="s", color="forestgreen", linewidth=2.5, label="OQD Defended Model")
    ax1.set_title("Top-1 Accuracy Under Rowhammer Attack", fontweight="bold")
    ax1.set_xlabel("Number of Targeted Bit Flips")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left")

    # Activation Outlier Squashing
    ax2.plot(flips, act_u_vals, marker="o", color="crimson", linewidth=2.5, label="Undefended (Outlier Explosion)")
    ax2.plot(flips, act_d_vals, marker="s", color="forestgreen", linewidth=2.5, label="OQD (Outlier Neutralized)")
    ax2.set_title("Max Activation Magnitude (Log Scale)", fontweight="bold")
    ax2.set_xlabel("Number of Targeted Bit Flips")
    ax2.set_ylabel("Max |Activation|")
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="center right")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n[+] Defense comparison plots saved to: {plot_path}")

if __name__ == "__main__":
    run_defense_comparison(max_flips=5)
