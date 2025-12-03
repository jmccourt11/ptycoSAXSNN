import torch
import torch.nn as nn

#SYMMETRY PENALTY
def NPCC_loss_symmetry_penalty(output, target):
    """
    Compute 1 - Pearson correlation coefficient as a loss function, with additional
    penalty for breaking inversion symmetry outside central 64x64 circle.
    Args:
        output: Predicted values (B, C, H, W)
        target: Target values (B, C, H, W)
    Returns:
        loss: Combined loss of correlation and symmetry terms
    """
    # Create circular mask for central 64x64 region
    h, w = output.shape[2:]
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
    center_y, center_x = h // 2, w // 2
    radius = 32  # 64/2 for 64x64 circle
    mask = ((y - center_y)**2 + (x - center_x)**2 <= radius**2).to(output.device)
    
    # Basic Pearson correlation loss
    output_flat = output.view(output.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    output_mean = output_flat.mean(dim=1, keepdim=True)
    target_mean = target_flat.mean(dim=1, keepdim=True)
    
    output_centered = output_flat - output_mean
    target_centered = target_flat - target_mean
    
    numerator = (output_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt((output_centered**2).sum(dim=1) * (target_centered**2).sum(dim=1))
    correlation = numerator / (denominator + 1e-8)
    pearson_loss = 1 - correlation.mean()
    
    # Symmetry loss for region outside circle
    output_flipped = torch.flip(output, dims=[-2, -1])  # Flip both height and width
    outside_mask = ~mask
    symmetry_loss = torch.mean(torch.abs(output[..., outside_mask] - output_flipped[..., outside_mask]))
    
    # Combine losses - can adjust weight of symmetry term
    symmetry_weight = 0.0
    total_loss = pearson_loss + symmetry_weight * symmetry_loss
    
    return total_loss, pearson_loss, symmetry_loss


def NPCC_loss(output, target):
    """
    Compute 1 - Pearson correlation coefficient as a loss function.
    Args:
        output: Predicted values (B, C, H, W)
        target: Target values (B, C, H, W)
    Returns:
        loss: 1 - correlation (to minimize)
    """
    # Basic Pearson correlation loss
    output_flat = output.view(output.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    output_mean = output_flat.mean(dim=1, keepdim=True)
    target_mean = target_flat.mean(dim=1, keepdim=True)
    
    output_centered = output_flat - output_mean
    target_centered = target_flat - target_mean
    
    numerator = (output_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt((output_centered**2).sum(dim=1) * (target_centered**2).sum(dim=1))
    correlation = numerator / (denominator + 1e-8)
    pearson_loss = 1 - correlation.mean()
    
    return pearson_loss


def L2_loss():
    #criterion = nn.MSELoss()
    return nn.MSELoss()

def L1_loss():
    #criterion = nn.L1Loss()
    return nn.L1Loss()