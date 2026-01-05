import torch
import torch.nn as nn

class STDPPlasticity(nn.Module):
    """
    Spike-Timing Dependent Plasticity rule.
    """
    def __init__(self, lr: float = 0.01, sigma: float = 0.1):
        super().__init__()
        self.lr = lr
        self.sigma = sigma

    def update_weights(self, weights: torch.Tensor, pre_activity: torch.Tensor, post_activity: torch.Tensor) -> torch.Tensor:
        """
        Simplified STDP update based on correlation of activities (Rate-based approximation).
        dw = lr * (post * pre - target)
        
        Args:
            weights: [Out, In]
            pre_activity: [B, In]
            post_activity: [B, Out]
        """
        # Ensure no NaNs
        if torch.isnan(weights).any():
            return weights

        # Calculate Hebbian term: Post * Pre
        # [B, Out, 1] * [B, 1, In] -> [B, Out, In]
        hebbian = torch.bmm(post_activity.unsqueeze(2), pre_activity.unsqueeze(1))
        delta_w = hebbian.mean(dim=0) # Average over batch
        
        # Soft bound regularization (simplification)
        new_weights = weights + self.lr * delta_w
        new_weights = torch.clamp(new_weights, 0.0, 1.0) # Excitatory only constraint
        
        return new_weights