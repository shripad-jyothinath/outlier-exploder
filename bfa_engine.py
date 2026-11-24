"""
bfa_engine.py
=============
Progressive Bit Search (PBS) Engine for Neural Network Fault Injection.

Based on:
- "Bit-Flip Attack: Crushing Neural Network with Progressive Bit Search" (Rakin et al., ICCV 2019)
- "GPUHammer: Rowhammer Attacks on GPU Memories are Practical" (Lin et al., USENIX Security 2025)

The algorithm leverages first-order gradient approximations to locate the
weights and specific bits whose inversion causes the steepest increase in loss:
    Delta L ≈ (dL / dw) * (w_flipped - w_original)
"""

import copy
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from bit_manipulation import flip_float32_bit, flip_float16_bit, analyze_flip_impact

class ProgressiveBitSearch:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        target_layers: Optional[List[str]] = None,
        dtype: str = "float32"
    ):
        self.model = model
        self.criterion = criterion
        self.dtype = dtype
        self.flip_history = []

        # Identify target layers (Linear and Conv2d weights)
        self.target_modules = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if target_layers is None or name in target_layers:
                    self.target_modules[name] = module

    def find_top_gradient_weights(self, top_k: int = 50) -> List[Dict]:
        """
        Screen across all target modules for weights with the highest gradient magnitude |dL/dw|.
        Returns a list of candidate coordinates.
        """
        candidates = []
        for name, module in self.target_modules.items():
            if module.weight.grad is None:
                continue
            grad = module.weight.grad.data.abs().view(-1)
            k = min(top_k, grad.numel())
            top_vals, top_indices = torch.topk(grad, k)

            weight_flat = module.weight.data.view(-1)
            for val, idx in zip(top_vals, top_indices):
                candidates.append({
                    "layer_name": name,
                    "module": module,
                    "flat_idx": idx.item(),
                    "grad_magnitude": val.item(),
                    "grad_val": module.weight.grad.data.view(-1)[idx].item(),
                    "current_weight": weight_flat[idx].item()
                })

        # Sort all candidates across all layers by gradient magnitude
        candidates.sort(key=lambda x: x["grad_magnitude"], reverse=True)
        return candidates[:top_k]

    def evaluate_bit_candidates(
        self,
        candidates: List[Dict],
        bits_to_test: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        For each top-gradient weight, evaluate the first-order loss shift:
            Delta L ≈ grad * (w_flipped - w)
        """
        if bits_to_test is None:
            # For float32: focus on Exponent bits [23..30] and MSB of mantissa [22]
            bits_to_test = list(range(20, 31))

        evaluated = []
        for cand in candidates:
            w_orig = cand["current_weight"]
            grad = cand["grad_val"]
            mod = cand["module"]

            for b in bits_to_test:
                if self.dtype == "float32":
                    w_flip = flip_float32_bit(w_orig, b)
                else:
                    w_flip = flip_float16_bit(w_orig, b)

                delta_w = w_flip - w_orig
                # First-order Taylor approximation of loss increase
                est_delta_loss = grad * delta_w

                # We want to MAXIMIZE loss (adversarial bit flip)
                evaluated.append({
                    "layer_name": cand["layer_name"],
                    "module": mod,
                    "flat_idx": cand["flat_idx"],
                    "bit_idx": b,
                    "w_orig": w_orig,
                    "w_flip": w_flip,
                    "delta_w": delta_w,
                    "est_delta_loss": est_delta_loss,
                    "segment": analyze_flip_impact(w_orig, b, self.dtype)["segment"]
                })

        # Sort descending by estimated loss increase
        evaluated.sort(key=lambda x: x["est_delta_loss"], reverse=True)
        return evaluated

    def execute_bit_flip(self, flip_info: Dict):
        """
        Physically mutate the target weight in model memory.
        """
        mod = flip_info["module"]
        flat_idx = flip_info["flat_idx"]
        w_flip = flip_info["w_flip"]

        with torch.no_grad():
            w_flat = mod.weight.data.view(-1)
            old_w = w_flat[flat_idx].item()
            w_flat[flat_idx] = w_flip

        record = {
            "step": len(self.flip_history) + 1,
            "layer_name": flip_info["layer_name"],
            "flat_idx": flat_idx,
            "bit_idx": flip_info["bit_idx"],
            "w_orig": old_w,
            "w_flip": w_flip,
            "delta_w": flip_info["delta_w"],
            "segment": flip_info["segment"],
            "est_delta_loss": flip_info["est_delta_loss"]
        }
        self.flip_history.append(record)
        return record

    def run_bfa_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        top_k_weights: int = 50,
        bits_to_test: Optional[List[int]] = None
    ) -> Dict:
        """
        Execute one complete Progressive Bit Search iteration:
        1. Zero gradients
        2. Forward pass & compute baseline loss
        3. Backward pass to obtain weight gradients
        4. Screen top weights and bit candidates
        5. Flip the single most impactful bit in model memory
        """
        self.model.eval()
        self.model.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()

        # Step A: Find highest gradient weights
        candidates = self.find_top_gradient_weights(top_k=top_k_weights)
        if not candidates:
            raise RuntimeError("No candidate weights found with non-zero gradients.")

        # Step B: Evaluate bit candidates
        evaluated_flips = self.evaluate_bit_candidates(candidates, bits_to_test=bits_to_test)
        if not evaluated_flips:
            raise RuntimeError("No valid bit candidates evaluated.")

        best_flip = evaluated_flips[0]

        # Step C: Physically flip the bit in memory
        record = self.execute_bit_flip(best_flip)
        return record
