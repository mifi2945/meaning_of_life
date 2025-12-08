"""
PyTorch-based parallel Game of Life simulation for CUDA acceleration.
This module provides batch processing capabilities for simulating multiple
Game of Life states simultaneously on GPU.
"""

import torch
import numpy as np
from typing import Optional, Tuple


def get_device() -> torch.device:
    """Get the best available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def numpy_to_torch(state: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert numpy array to PyTorch tensor on specified device."""
    if device is None:
        device = get_device()
    return torch.from_numpy(state).to(device).float()


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to numpy array."""
    return tensor.cpu().numpy().astype(np.uint8)


def batch_simulate(
    initial_states: torch.Tensor,
    steps: int,
    expansion: int = -1,
    include_between: bool = False,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Batch simulate multiple Game of Life states in parallel.
    
    Args:
        initial_states: Tensor of shape (batch_size, state_size) where state_size is a perfect square
        steps: Number of simulation steps
        expansion: Grid expansion size (-1 for infinite)
        include_between: Whether to return intermediate states
        device: Device to run on (defaults to best available)
        
    Returns:
        Final states tensor of shape (batch_size, grid_h, grid_w)
        If include_between, returns (batch_size, steps+1, grid_h, grid_w)
    """
    if device is None:
        device = get_device()
    
    batch_size = initial_states.shape[0]
    state_size = initial_states.shape[1]
    s = int(state_size ** 0.5)
    
    # Calculate grid size
    if expansion == -1:
        size = s + (steps // 2) + 1
    else:
        size = s + expansion
    
    # Initialize batch grid
    grids = torch.zeros((batch_size, size, size), dtype=torch.float32, device=device)
    
    # Place initial states in center
    middle = size // 2
    half = s // 2
    bounds = (middle - half, middle + half + (s % 2))
    
    # Reshape initial states and place in grids
    initial_grids = initial_states.view(batch_size, s, s)
    grids[:, bounds[0]:bounds[1], bounds[0]:bounds[1]] = initial_grids
    
    # Neighbor counting kernel (8 neighbors)
    kernel = torch.tensor([[1., 1., 1.],
                           [1., 0., 1.],
                           [1., 1., 1.]], device=device).unsqueeze(0).unsqueeze(0)
    
    # Track bounding boxes for each batch item
    min_rs = torch.full((batch_size,), bounds[0], dtype=torch.long, device=device)
    max_rs = torch.full((batch_size,), bounds[1], dtype=torch.long, device=device)
    min_cs = torch.full((batch_size,), bounds[0], dtype=torch.long, device=device)
    max_cs = torch.full((batch_size,), bounds[1], dtype=torch.long, device=device)
    
    if include_between:
        log = [grids.clone()]
    
    # Simulate steps
    for step in range(steps):
        # Find active regions for each batch item
        # We'll process all items, but optimize by only updating active regions
        new_grids = grids.clone()
        
        # For each batch item, find the region to update
        for b in range(batch_size):
            min_r = max(0, min_rs[b].item() - 1)
            max_r = min(size, max_rs[b].item() + 1)
            min_c = max(0, min_cs[b].item() - 1)
            max_c = min(size, max_cs[b].item() + 1)
            
            if min_r >= max_r or min_c >= max_c:
                continue
                
            region = grids[b:b+1, min_r:max_r, min_c:max_c].unsqueeze(0)  # (1, 1, h, w)
            
            # Count neighbors using convolution
            neighbors = torch.nn.functional.conv2d(
                region, kernel, padding=1
            ).squeeze(0).squeeze(0)  # (h, w)
            
            # Game of Life rules
            survives = (region.squeeze(0).squeeze(0) == 1) & ((neighbors == 2) | (neighbors == 3))
            born = (region.squeeze(0).squeeze(0) == 0) & (neighbors == 3)
            new_region = (survives | born).float()
            
            new_grids[b, min_r:max_r, min_c:max_c] = new_region
            
            # Update bounding box
            alive = torch.nonzero(new_region, as_tuple=False)
            if len(alive) > 0:
                min_rs[b] = max(0, alive[:, 0].min().item() + min_r - 1)
                max_rs[b] = min(size, alive[:, 0].max().item() + min_r + 2)
                min_cs[b] = max(0, alive[:, 1].min().item() + min_c - 1)
                max_cs[b] = min(size, alive[:, 1].max().item() + min_c + 2)
            else:
                # All dead, keep same bounds
                pass
        
        grids = new_grids
        
        if include_between:
            log.append(grids.clone())
    
    if include_between:
        return torch.stack(log, dim=1)  # (batch_size, steps+1, h, w)
    else:
        return grids  # (batch_size, h, w)


def batch_simulate_optimized(
    initial_states: torch.Tensor,
    steps: int,
    expansion: int = -1,
    include_between: bool = False,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Optimized batch simulation using full-grid convolution.
    This version is faster for large batches but uses more memory.
    
    Args:
        initial_states: Tensor of shape (batch_size, state_size)
        steps: Number of simulation steps
        expansion: Grid expansion size (-1 for infinite)
        include_between: Whether to return intermediate states
        device: Device to run on
        
    Returns:
        Final states tensor
    """
    if device is None:
        device = get_device()
    
    batch_size = initial_states.shape[0]
    state_size = initial_states.shape[1]
    s = int(state_size ** 0.5)
    
    # Calculate grid size
    if expansion == -1:
        size = s + (steps // 2) + 1
    else:
        size = s + expansion
    
    # Initialize batch grid
    grids = torch.zeros((batch_size, size, size), dtype=torch.float32, device=device)
    
    # Place initial states in center
    middle = size // 2
    half = s // 2
    bounds = (middle - half, middle + half + (s % 2))
    
    initial_grids = initial_states.view(batch_size, s, s)
    grids[:, bounds[0]:bounds[1], bounds[0]:bounds[1]] = initial_grids
    
    # Neighbor counting kernel - needs to be (out_channels, in_channels, h, w)
    kernel = torch.tensor([[1., 1., 1.],
                           [1., 0., 1.],
                           [1., 1., 1.]], device=device).unsqueeze(0).unsqueeze(0)
    
    if include_between:
        log = [grids.clone()]
    
    # Simulate steps - process entire batch at once
    for step in range(steps):
        # Add channel dimension for conv2d: (batch_size, 1, h, w)
        grids_batched = grids.unsqueeze(1)
        
        # Count neighbors for all grids at once
        # conv2d expects (batch, channels, h, w) and kernel (out_ch, in_ch, h, w)
        neighbors = torch.nn.functional.conv2d(
            grids_batched, kernel, padding=1
        ).squeeze(1)  # (batch_size, h, w)
        
        # Game of Life rules
        survives = (grids == 1) & ((neighbors == 2) | (neighbors == 3))
        born = (grids == 0) & (neighbors == 3)
        grids = (survives | born).float()
        
        if include_between:
            log.append(grids.clone())
    
    if include_between:
        return torch.stack(log, dim=1)
    else:
        return grids


def batch_value_growth(
    final_states: torch.Tensor,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Batch evaluate growth problem fitness for multiple final states.
    
    Args:
        final_states: Tensor of shape (batch_size, grid_h, grid_w)
        device: Device to run on
        
    Returns:
        Fitness scores tensor of shape (batch_size,)
    """
    if device is None:
        device = get_device()
    
    # Count alive cells
    alive = final_states.sum(dim=(1, 2))  # (batch_size,)
    total = final_states.shape[1] * final_states.shape[2]
    
    # Handle dead states
    mask = alive > 0
    scores = torch.zeros_like(alive)
    
    if not mask.any():
        return scores
    
    # Neighbor counting kernel for clump score
    kernel = torch.tensor([[1., 1., 1.],
                           [1., 0., 1.],
                           [1., 1., 1.]], device=device).unsqueeze(0).unsqueeze(0)
    
    # Count neighbors for all grids
    states_batched = final_states.unsqueeze(1)  # (batch_size, 1, h, w)
    neighbor_counts = torch.nn.functional.conv2d(
        states_batched, kernel, padding=1
    ).squeeze(1)  # (batch_size, h, w)
    
    # Calculate clump score: average neighbors per alive cell
    clump_scores = (neighbor_counts * final_states).sum(dim=(1, 2)) / (alive * 8 + 1e-8)
    clump_scores = torch.clamp(clump_scores, 0, 1)
    
    # Density
    density = alive / total
    
    # Weighted score
    scores[mask] = density[mask] * (0.5 + 0.5 * clump_scores[mask])
    
    return scores

