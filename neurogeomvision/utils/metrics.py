import torch
import numpy as np
from typing import Dict, List

class CorticalMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0
        self.activations_sum = 0.0
        self.batches = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, internals: Dict):
        # Accuracy
        preds = torch.argmax(logits, dim=1)
        self.correct += (preds == targets).sum().item()
        self.total += targets.size(0)

        # Energy Proxy (L1 norm of ventral features)
        if 'ventral_features' in internals:
            self.activations_sum += internals['ventral_features'].abs().mean().item()
        self.batches += 1

    def get_results(self) -> Dict[str, float]:
        acc = self.correct / self.total if self.total > 0 else 0.0
        energy = self.activations_sum / self.batches if self.batches > 0 else 0.0
        return {
            "accuracy_top1": acc,
            "mean_activation_energy": energy
        }