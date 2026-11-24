"""
run_experiment.py
=================
End-to-End Evaluation of Progressive Bit-Flip Attack (BFA) on Neural Networks.

Demonstrating:
1. The Outlier Amplifier Mechanism: How an exponent bit flip causes weights to explode.
2. Fast progressive bit search based on first-order Taylor expansion of the loss function.
3. The Winner-Group Effect: How corrupted networks collapse to predicting a single class.
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from tabulate import tabulate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bit_manipulation import analyze_flip_impact
from bfa_engine import ProgressiveBitSearch
from activation_monitor import ActivationMonitor, compute_prediction_entropy, check_winner_group_collapse

def generate_evaluation_dataset(model: nn.Module, batch_size: int = 64, num_batches: int = 4):
    """
    Generate deterministic synthetic image batches labeled by the clean model's predictions.
    This guarantees clean baseline accuracy is exactly 100.0%, providing a precise baseline
    to measure relative accuracy drop (RAD) and degradation curves.
    """
    torch.manual_seed(42)
    dataset = []
    model.eval()
    with torch.no_grad():
        for _ in range(num_batches):
            x = torch.randn(batch_size, 3, 224, 224)
            logits = model(x)
            y = torch.argmax(logits, dim=-1)
            dataset.append((x, y))
    return dataset

def evaluate_model(model: nn.Module, dataset, criterion: nn.Module):
    """Evaluate accuracy, loss, and class prediction statistics across the dataset."""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for x, y in dataset:
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            total_correct += (preds == y).sum().item()
            total_samples += len(y)
            all_preds.append(preds)
            all_probs.append(probs)

    accuracy = (total_correct / total_samples) * 100.0
    avg_loss = total_loss / total_samples
    all_preds = torch.cat(all_preds, dim=0)
    all_probs = torch.cat(all_probs, dim=0)

    entropy = compute_prediction_entropy(all_probs)
    collapse_info = check_winner_group_collapse(all_preds)

    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "entropy": entropy,
        "collapse": collapse_info
    }

def run_attack_sweep(model, dataset, criterion, monitor, mode="exponent", max_flips=5):
    """
    Execute attack sweep under specified regime:
      - 'exponent': Progressive Bit Search on Exponent bits [23..30] (Outlier Amplifier)
      - 'mantissa': Progressive Bit Search on Mantissa bits [15..22] (Progressive Degradation)
      - 'random': Random bit flips across target layers (Hardware Noise Baseline)
    """
    calib_x, calib_y = dataset[0]
    bfa = ProgressiveBitSearch(
        model=model,
        criterion=criterion,
        target_layers=["layer4.1.conv2", "fc"]
    )

    bits_pool = list(range(23, 31)) if mode == "exponent" else list(range(15, 23))
    history = []
    
    # Baseline
    monitor.clear()
    base_eval = evaluate_model(model, dataset, criterion)
    act_sum = monitor.get_summary()
    history.append({
        "flip": 0,
        "mode": mode,
        "layer": "Clean",
        "bit": "-",
        "accuracy": base_eval["accuracy"],
        "loss": base_eval["loss"],
        "max_activation": min(act_sum.get("max_val", 0.0), 1e38),
        "winner_ratio": base_eval["collapse"]["winner_ratio"]
    })

    print(f"\n---> Running Attack Regime: [{mode.upper()}] (up to {max_flips} flips)...")

    for step in range(1, max_flips + 1):
        if mode in ["exponent", "mantissa"]:
            record = bfa.run_bfa_step(calib_x, calib_y, top_k_weights=20, bits_to_test=bits_pool)
        else: # random
            # Pick random layer and random coordinate
            layer_name = np.random.choice(list(bfa.target_modules.keys()))
            mod = bfa.target_modules[layer_name]
            flat_idx = np.random.randint(0, mod.weight.numel())
            bit_idx = np.random.randint(0, 32)
            w_orig = mod.weight.data.view(-1)[flat_idx].item()
            from bit_manipulation import flip_float32_bit
            w_flip = flip_float32_bit(w_orig, bit_idx)
            record = bfa.execute_bit_flip({
                "module": mod,
                "layer_name": layer_name,
                "flat_idx": flat_idx,
                "bit_idx": bit_idx,
                "w_flip": w_flip,
                "delta_w": w_flip - w_orig,
                "segment": f"Random bit {bit_idx}",
                "est_delta_loss": 0.0
            })

        monitor.clear()
        eval_stats = evaluate_model(model, dataset, criterion)
        act_sum = monitor.get_summary()

        acc = eval_stats["accuracy"]
        loss = min(eval_stats["loss"], 1e6) if not np.isnan(eval_stats["loss"]) else 1e6
        max_act = min(act_sum.get("max_val", 0.0), 1e38)

        print(f"  Step {step}: {record['layer_name'].split('.')[-1]} bit {record['bit_idx']} | Acc: {acc:.1f}% | Loss: {loss:.2f} | MaxAct: {max_act:.2e}")

        history.append({
            "flip": step,
            "mode": mode,
            "layer": record["layer_name"],
            "bit": record["bit_idx"],
            "accuracy": acc,
            "loss": loss,
            "max_activation": max_act,
            "winner_ratio": eval_stats["collapse"]["winner_ratio"]
        })

        if acc == 0.0 and eval_stats["collapse"]["winner_ratio"] >= 1.0:
            print(f"  [!] Model collapsed completely at flip {step}.")
            break

    return history
    parser = argparse.ArgumentParser(description="Progressive Bit-Flip Attack (BFA) Evaluation")
    parser.add_argument("--max-flips", type=int, default=5, help="Maximum bit flips to execute")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num-batches", type=int, default=2, help="Number of evaluation batches")
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory for output charts and logs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 80)
    print(" PROJECT 01: THE OUTLIER EXPLODER — PROGRESSIVE BIT-FLIP ATTACK (BFA)")
    print(" Evaluating Neural Network Reliability Limits under Hardware Memory Faults")
    print("=" * 80)

    # 1. Load Pretrained Model
    print("[1/4] Loading pretrained ResNet-18 model...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # 2. Attach Activation Monitor
    monitor = ActivationMonitor(model)

    # 3. Generate Evaluation Data
    print(f"[2/4] Generating evaluation dataset ({args.num_batches} batches of {args.batch_size} samples)...")
    dataset = generate_evaluation_dataset(batch_size=args.batch_size, num_batches=args.num_batches)
    calib_x, calib_y = dataset[0]  # First batch used for gradient-based bit search

    # 4. Measure Baseline Performance
    print("[3/4] Measuring clean baseline performance...")
    monitor.clear()
    baseline_stats = evaluate_model(model, dataset, criterion)
    act_summary = monitor.get_summary()

    print(f"  -> Clean Baseline Accuracy : {baseline_stats['accuracy']:.2f}%")
    print(f"  -> Clean Cross-Entropy Loss: {baseline_stats['loss']:.4f}")
    print(f"  -> Prediction Entropy      : {baseline_stats['entropy']:.4f}")
    print(f"  -> Max Baseline Activation : {act_summary.get('max_val', 0.0):.4f} (layer: {act_summary.get('max_layer_name', '')})\n")

    # 5. Initialize BFA Engine
    bfa = ProgressiveBitSearch(
        model=model,
        criterion=criterion,
        target_layers=["layer4.1.conv2", "fc"]  # High-impact deep feature & classifier layers
    )

    # Tracking log
    history = [{
        "flip": 0,
        "layer": "N/A (Clean)",
        "bit": "-",
        "segment": "Clean",
        "w_orig": 0.0,
        "w_flip": 0.0,
        "delta_w": 0.0,
        "accuracy": baseline_stats["accuracy"],
        "loss": baseline_stats["loss"],
        "max_activation": act_summary.get("max_val", 0.0),
        "entropy": baseline_stats["entropy"],
        "winner_ratio": baseline_stats["collapse"]["winner_ratio"]
    }]

    print("[4/4] Executing Progressive Bit Search (PBS) iterations...")
    table_headers = ["Flip #", "Target Layer", "Bit", "Orig W", "Flipped W", "Accuracy", "Loss", "Max Activation", "Winner %"]
    table_rows = []

    for step in range(1, args.max_flips + 1):
        t0 = time.time()
        # Find and execute single most damaging bit flip
        record = bfa.run_bfa_step(calib_x, calib_y, top_k_weights=20, bits_to_test=[30, 29, 28, 27, 26, 25, 24, 23, 22])

        # Evaluate model after the flip
        monitor.clear()
        eval_stats = evaluate_model(model, dataset, criterion)
        act_summary = monitor.get_summary()
        elapsed = time.time() - t0

        row = [
            step,
            record["layer_name"].split(".")[-1],
            f"b{record['bit_idx']}",
            f"{record['w_orig']:+.4f}",
            f"{record['w_flip']:+.2e}",
            f"{eval_stats['accuracy']:.2f}%",
            f"{eval_stats['loss']:.2f}",
            f"{act_summary.get('max_val', 0.0):.2e}",
            f"{eval_stats['collapse']['winner_ratio']*100:.1f}%"
        ]
        table_rows.append(row)

        history.append({
            "flip": step,
            "layer": record["layer_name"],
            "bit": record["bit_idx"],
            "segment": record["segment"],
            "w_orig": record["w_orig"],
            "w_flip": record["w_flip"],
            "delta_w": record["delta_w"],
            "accuracy": eval_stats["accuracy"],
            "loss": eval_stats["loss"],
            "max_activation": act_summary.get("max_val", 0.0),
            "entropy": eval_stats["entropy"],
            "winner_ratio": eval_stats["collapse"]["winner_ratio"]
        })

        print(f"  -> Flip {step}: {record['layer_name']} bit {record['bit_idx']} | Acc: {eval_stats['accuracy']:.2f}% | MaxAct: {act_summary.get('max_val', 0.0):.2e} ({elapsed:.2f}s)")

        # Early termination if accuracy is completely crushed (0.0%) and collapsed
        if eval_stats["accuracy"] <= 0.0 and eval_stats["collapse"]["winner_ratio"] >= 1.0:
            print("  [!] Complete model collapse achieved (0.0% accuracy, 100% winner class saturation).")
            break

def main():
    parser = argparse.ArgumentParser(description="Progressive Bit-Flip Attack (BFA) Evaluation")
    parser.add_argument("--max-flips", type=int, default=5, help="Maximum bit flips to execute")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num-batches", type=int, default=2, help="Number of evaluation batches")
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory for output charts and logs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 80)
    print(" PROJECT 01: THE OUTLIER EXPLODER — PROGRESSIVE BIT-FLIP ATTACK (BFA)")
    print(" Comparing Targeted Exponent vs. Mantissa vs. Random Hardware Memory Faults")
    print("=" * 80)

    criterion = nn.CrossEntropyLoss()

    # Pre-generate evaluation dataset using fresh ResNet-18
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
    print(f"Generating evaluation dataset ({args.num_batches} batches of {args.batch_size} samples)...")
    dataset = generate_evaluation_dataset(base_model, batch_size=args.batch_size, num_batches=args.num_batches)

    results = {}
    for mode in ["exponent", "mantissa", "random"]:
        # Load fresh pretrained weights for each test
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
        mon = ActivationMonitor(m)
        hist = run_attack_sweep(m, dataset, criterion, mon, mode=mode, max_flips=args.max_flips)
        mon.remove()
        results[mode] = hist

    # Save results to JSON
    json_path = os.path.join(args.output_dir, "bfa_comparative_log.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results logged to: {json_path}")

    # Print summary table
    summary_rows = []
    max_steps = max(len(results[m]) for m in results)
    for s in range(max_steps):
        row = [s]
        for m in ["exponent", "mantissa", "random"]:
            if s < len(results[m]):
                row.append(f"{results[m][s]['accuracy']:.1f}%")
            else:
                row.append(f"{results[m][-1]['accuracy']:.1f}%*")
        summary_rows.append(row)

    print("\n" + tabulate(summary_rows, headers=["Flip #", "Targeted Exponent (PBS)", "Targeted Mantissa (PBS)", "Random Faults"], tablefmt="github"))

    # Generate Comparative Visualization Plot
    plot_path = os.path.join(args.output_dir, "bfa_comparative_curves.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Neural Network Vulnerability to Targeted vs. Random Memory Bit Flips", fontsize=14, fontweight="bold")

    styles = {
        "exponent": {"label": "Targeted Exponent (PBS)", "color": "crimson", "marker": "o"},
        "mantissa": {"label": "Targeted Mantissa (PBS)", "color": "dodgerblue", "marker": "s"},
        "random": {"label": "Random Hardware Faults", "color": "gray", "marker": "^"}
    }

    for m, style in styles.items():
        hist = results[m]
        flips = [h["flip"] for h in hist]
        accs = [h["accuracy"] for h in hist]
        acts = [h["max_activation"] for h in hist]

        ax1.plot(flips, accs, label=style["label"], color=style["color"], marker=style["marker"], linewidth=2)
        ax2.plot(flips, acts, label=style["label"], color=style["color"], marker=style["marker"], linewidth=2)

    ax1.set_title("Top-1 Accuracy Drop vs. Bit Flips", fontweight="bold")
    ax1.set_xlabel("Number of Injected Bit Flips")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    ax2.set_title("Max Layer Activation (Outlier Explosion)", fontweight="bold")
    ax2.set_xlabel("Number of Injected Bit Flips")
    ax2.set_ylabel("Max |Activation| (Log Scale)")
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[+] Comparative diagnostic plots saved to: {plot_path}")

if __name__ == "__main__":
    main()
