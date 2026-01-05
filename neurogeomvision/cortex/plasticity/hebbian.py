import torch
import torch.nn as nn

class HebbianPlasticity(nn.Module):
    def __init__(self, learning_rate: float = 1e-4):
        super().__init__()
        self.lr = learning_rate

    def update_weights(self, w: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Oja's Rule: dw = lr * y * (x - y * w)
        Stabilizes Hebbian learning.
        """
        batch_size = x.size(0)
        # y: [B, Out], x: [B, In], w: [Out, In]
        
        # y * x -> [B, Out, In]
        hebbian_term = torch.bmm(y.unsqueeze(2), x.unsqueeze(1)).mean(dim=0)
        
        # y^2 * w -> [Out, In] (approx)
        y_squared = (y ** 2).mean(dim=0).unsqueeze(1) # [Out, 1]
        forgetting_term = y_squared * w
        
        delta_w = self.lr * (hebbian_term - forgetting_term)
        return w + delta_w

class HomeostaticPlasticity(nn.Module):
    def __init__(self, target_activity: float = 0.1, lr: float = 1e-3):
        super().__init__()
        self.target = target_activity
        self.lr = lr

    def update_weights(self, w: torch.Tensor, activity: torch.Tensor) -> torch.Tensor:
        """
        Scales weights to maintain target activity.
        """
        current_mean = activity.mean()
        # If activity > target, decrease weights.
        factor = 1.0 + self.lr * (self.target - current_mean)
        return w * factor