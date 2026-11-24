"""
activation_monitor.py
=====================
Forward hooks and statistics tracking to visualize the "Outlier Amplifier Mechanism".

When an MSB exponent bit flip occurs in layer L, the corrupted weight creates
an explosive activation value that overwhelms legitimate representations and
causes the network's predictions to collapse into a single degenerate class
(the "Winner-Group Effect").
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List

class ActivationMonitor:
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.layer_stats = {}

        # Register forward hooks on all linear layers and conv layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            # output shape: [batch, channels, H, W] or [batch, features]
            data = output.detach()
            max_val = data.abs().max().item()
            mean_val = data.abs().mean().item()
            std_val = data.std().item() if data.numel() > 1 else 0.0

            self.layer_stats[name] = {
                "max_activation": max_val,
                "mean_activation": mean_val,
                "std_activation": std_val,
                "has_inf_nan": bool(torch.isnan(data).any() or torch.isinf(data).any())
            }
        return hook_fn

    def clear(self):
        self.layer_stats.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_summary(self) -> Dict:
        """Find the layer with the largest activation outlier across the model."""
        if not self.layer_stats:
            return {}
        max_layer = max(self.layer_stats.items(), key=lambda x: x[1]["max_activation"])
        return {
            "max_layer_name": max_layer[0],
            "max_val": max_layer[1]["max_activation"],
            "mean_val": max_layer[1]["mean_activation"],
            "has_inf_nan": max_layer[1]["has_inf_nan"],
            "all_layers": self.layer_stats
        }

def compute_prediction_entropy(probs: torch.Tensor) -> float:
    """
    Compute mean Shannon entropy across predicted batch probabilities:
        H(p) = - sum(p * log(p + 1e-12))
    """
    p = probs + 1e-12
    entropy = - (p * torch.log(p)).sum(dim=-1).mean().item()
    return entropy

def check_winner_group_collapse(predictions: torch.Tensor) -> Dict:
    """
    Analyze whether predictions across the evaluation set have collapsed to a single class.
    """
    preds = predictions.cpu().numpy()
    unique, counts = np.unique(preds, return_counts=True)
    top_class = unique[np.argmax(counts)]
    top_ratio = np.max(counts) / len(preds)
    is_collapsed = top_ratio > 0.8  # >80% of all samples predict the exact same class

    return {
        "is_collapsed": is_collapsed,
        "winner_class": int(top_class),
        "winner_ratio": float(top_ratio),
        "unique_classes_predicted": len(unique)
    }
