import torch
import torchvision
import os

def save_activation_grid(tensor: torch.Tensor, path: str, normalize: bool = True):
    """
    Saves a grid of feature maps from a tensor [B, C, H, W].
    Takes the first item in batch.
    """
    if tensor.dim() != 4:
        return
    
    # Take first image in batch: [C, H, W]
    first_img_activations = tensor[0].unsqueeze(1) # [C, 1, H, W]
    
    # Limit to first 64 channels to avoid huge images
    if first_img_activations.shape[0] > 64:
        first_img_activations = first_img_activations[:64]

    grid = torchvision.utils.make_grid(first_img_activations, normalize=normalize, nrow=8)
    
    # Convert to PIL
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    
    from PIL import Image
    im = Image.fromarray(ndarr)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path)