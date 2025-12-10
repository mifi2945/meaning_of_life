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
    Note: This still processes individually due to the complexity of canonicalization.
    
    Args:
        final_states: Tensor of shape (batch_size, h, w)
        device: PyTorch device
    
    Returns:
        List of descriptor arrays
    """
    if device is None:
        device = get_device()
    
    batch_size = final_states.shape[0]
    descriptors = []
    
    # Convert to numpy for processing (can be optimized further)
    states_np = final_states.cpu().numpy()
    
    for i in range(batch_size):
        state = states_np[i]
        rows, cols = np.where(state == 1)
        if len(rows) == 0:
            descriptors.append(np.zeros((1,), dtype=np.uint8))
            continue
        
        rmin, rmax = rows.min(), rows.max()
        cmin, cmax = cols.min(), cols.max()
        
        shape = state[rmin:rmax+1, cmin:cmax+1]
        
        # Generate all 8 isometries
        transforms = [
            shape,
            np.rot90(shape, 1),
            np.rot90(shape, 2),
            np.rot90(shape, 3),
            np.fliplr(shape),
            np.flipud(shape),
            np.rot90(np.fliplr(shape), 1),
            np.rot90(np.flipud(shape), 1),
        ]
        
        # Pick canonical representation
        canonical = min(t.flatten().tobytes() for t in transforms)
        descriptors.append(np.frombuffer(canonical, dtype=np.uint8))
    
    return descriptors


def _to_index_set(arr: np.ndarray) -> set:
    """Convert array to set of indices (helper for novelty calculation)."""
    a = np.asarray(arr)
    if a.size == 0:
        return set()
    if a.ndim != 1:
        a = a.ravel()
    nz = np.nonzero(a)[0]
    return set(map(int, nz))


def _jaccard_set(a: set, b: set) -> float:
    """Compute Jaccard distance between two sets (helper for novelty calculation)."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return 1.0 - (inter / union)


def novelty_batch(
    descriptors: List[np.ndarray],
    archive: List[np.ndarray],
    k: int = 10,
    device: torch.device = None
) -> np.ndarray:
    """
    Compute novelty scores for a batch of descriptors.
    Uses Jaccard distance for efficiency.
    
    Args:
        descriptors: List of descriptor arrays
        archive: List of archive descriptor arrays
        k: Number of nearest neighbors
        device: PyTorch device (unused, kept for API compatibility)
    
    Returns:
        Array of novelty scores
    """
    novelties = []
    
    for desc in descriptors:
        # Convert primary descriptor -> index set
        desc_set = _to_index_set(desc)
        
        # Build pool of sets (population followed by archive)
        pool_arrays = list(descriptors) + list(archive)
        pool_sets = [_to_index_set(p) for p in pool_arrays]
        
        # Compute distances but exclude exactly one self-match (if present).
        dists = []
        self_skipped = False
        for other in pool_sets:
            if (other == desc_set) and (not self_skipped):
                # skip the first exact-equal match (treating that as self)
                self_skipped = True
                continue
            d = _jaccard_set(desc_set, other)
            dists.append(d)
        
        if len(dists) == 0:
            # No neighbors to compare to
            novelty = 0.0
        else:
            # Use up to k nearest neighbors
            k_eff = min(k, len(dists))
            dists.sort()
            novelty = float(sum(dists[:k_eff]) / k_eff)
        
        novelties.append(novelty)
    
    return np.array(novelties)

