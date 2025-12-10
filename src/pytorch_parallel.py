# Usa da Cuda
import torch
import numpy as np
from typing import List, Tuple
from problems import Parameters


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def simulate_batch(
    initial_states,
    parameters: Parameters,
    device: torch.device = None
) -> torch.Tensor:
    if device is None:
        device = get_device()
    
    batch_size = initial_states.shape[0]
    state_size = initial_states.shape[1]
    steps = parameters.steps
    expansion = parameters.expansion
    
    # Dimension of the initial grid
    s = int(state_size ** 0.5)
    
    # Grid size calculation
    size = s + (steps // 2) + 1 if expansion == -1 else s + expansion
    
    # Convert to torch tensors
    if isinstance(initial_states, np.ndarray):
        M = torch.zeros((batch_size, size, size), dtype=torch.float32, device=device)
        initial_tensor = torch.from_numpy(initial_states).to(device)
    else:
        M = torch.zeros((batch_size, size, size), dtype=torch.float32, device=device)
        initial_tensor = initial_states.to(device)
    
    # Place initial states in center of grids
    middle = size // 2
    half = s // 2
    bounds = (middle - half, middle + half + (s % 2))
    
    # Reshape and place initial states
    initial_grids = initial_tensor.view(batch_size, s, s).float()
    M[:, bounds[0]:bounds[1], bounds[0]:bounds[1]] = initial_grids
    
    # Kernel for counting neighbors (8 neighbors around each cell)
    # Shape: (1, 1, 3, 3) for batch conv2d
    kernel = torch.tensor([[1., 1., 1.],
                           [1., 0., 1.],
                           [1., 1., 1.]], device=device).view(1, 1, 3, 3)
    
    # Process all steps in parallel across the batch
    for step in range(steps):
        # Check if any batch items have living cells
        if not torch.any(M > 0):
            break
        
        # Pad M for convolution (batch, channels, height, width)
        M_padded = M.unsqueeze(1)  # (batch_size, 1, h, w)
        
        # Count neighbors for all batches in parallel
        neighbors = torch.nn.functional.conv2d(
            M_padded, kernel, padding=1
        ).squeeze(1)  # (batch_size, h, w)
        
        # Game of Life rules - vectorized across entire batch
        survives = (M == 1.0) & ((neighbors == 2.0) | (neighbors == 3.0))
        born = (M == 0.0) & (neighbors == 3.0)
        M = (survives | born).float()
    
    # Convert back to uint8 for consistency
    return M.byte()


def value_batch(
    final_states: torch.Tensor,
    problem_type: str = "growth"
) -> torch.Tensor:
    """
    Batched fitness evaluation.
    
    Args:
        final_states: Tensor of shape (batch_size, h, w)
        problem_type: Type of problem ("growth" or "migration")
    
    Returns:
        Fitness values as tensor of shape (batch_size,)
    """
    if problem_type == "growth":
        # Count alive cells for each state
        return final_states.sum(dim=(1, 2)).float()
    elif problem_type == "migration":
        # Placeholder for migration problem
        return torch.ones(final_states.shape[0], device=final_states.device)
    else:
        return final_states.sum(dim=(1, 2)).float()


def behavior_descriptor_batch(
    final_states: torch.Tensor,
    device: torch.device = None
) -> List[np.ndarray]:
    """
    Compute behavior descriptors for a batch of final states.
    Matches the CGOL_Problem.behavior_descriptor implementation.
    
    Args:
        final_states: Tensor of shape (batch_size, h, w)
        device: PyTorch device
    
    Returns:
        List of descriptor arrays (feature vectors: [population/100, width/20, height/20, density])
    """
    if device is None:
        device = get_device()
    
    batch_size = final_states.shape[0]
    descriptors = []
    
    # Convert to numpy for processing
    states_np = final_states.cpu().numpy()
    
    for i in range(batch_size):
        state = states_np[i]
        rows, cols = np.where(state == 1)
        
        if len(rows) == 0:
            descriptors.append(np.array([0.0, 0.0, 0.0, 0.0]))
            continue
        
        # Calculate Bounding Box
        rmin, rmax = rows.min(), rows.max()
        cmin, cmax = cols.min(), cols.max()
        
        height = rmax - rmin + 1
        width = cmax - cmin + 1
        population = len(rows)
        area = height * width
        density = population / area if area > 0 else 0
        
        # Normalize large values to keep distances balanced
        descriptors.append(np.array([
            population / 100.0,  # Normalize population
            width / 20.0,        # Normalize width
            height / 20.0,       # Normalize height
            density              # Density is already 0-1
        ]))
    
    return descriptors


def novelty_batch(
    descriptors: List[np.ndarray],
    archive: List[np.ndarray],
    k: int = 10,
    device: torch.device = None
) -> np.ndarray:
    """
    Compute novelty scores for a batch of descriptors.
    Uses Euclidean distance to match CGOL_Problem.novelty implementation.
    
    Args:
        descriptors: List of descriptor arrays (feature vectors)
        archive: List of archive descriptor arrays
        k: Number of nearest neighbors
        device: PyTorch device (unused, kept for API compatibility)
    
    Returns:
        Array of novelty scores
    """
    from scipy.spatial.distance import cdist
    
    novelties = []
    
    for desc in descriptors:
        # Combine archive and current population for the comparison pool
        pool = np.array(archive + descriptors)
        
        if len(pool) == 0:
            novelties.append(0.0)
            continue
        
        # Calculate Euclidean distances from 'desc' to everyone in 'pool'
        dists = cdist(desc.reshape(1, -1), pool, metric='euclidean')[0]
        
        # Sort distances
        dists_sorted = np.sort(dists)
        
        # Filter out zero distances (self-matches) - take first k non-zero if available
        non_zero_dists = dists_sorted[dists_sorted > 1e-10]
        
        if len(non_zero_dists) == 0:
            # All distances are zero (only self in pool)
            novelty = 0.0
        else:
            # Use up to k nearest neighbors
            k_eff = min(k, len(non_zero_dists))
            novelty = float(np.mean(non_zero_dists[:k_eff]))
        
        novelties.append(novelty)
    
    return np.array(novelties)

