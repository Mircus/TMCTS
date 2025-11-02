
from __future__ import annotations
import numpy as np
import json
from typing import List, Dict, Tuple, Union
import random
from pathlib import Path

def generate_task(grid_size: Tuple[int, int], alphabet: Tuple[int, ...], missing: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a single ARC task with exact paper specifications."""
    rng = np.random.default_rng(seed)
    n, m = grid_size
    
    # Generate target grid with symbols from alphabet
    target = rng.integers(low=alphabet[0], high=alphabet[-1]+1, size=(n, m))
    initial = target.copy()
    
    # Select missing positions randomly
    all_pos = [(i, j) for i in range(n) for j in range(m)]
    rng.shuffle(all_pos)
    for (i, j) in all_pos[:missing]:
        initial[i, j] = -1
    
    return initial, target

def compute_depth_bound(missing_count: int) -> int:
    """Compute depth bound as per paper: dmax = 2*(#missing)+1"""
    return 2 * missing_count + 1

def generate_dataset_spec(
    num_tasks: int = 50, 
    seed: int = 42,
    grid_mix: Tuple[float, float, float] = (0.60, 0.25, 0.15),  # 3x3, 4x4, 5x5
    alphabet: Tuple[int, ...] = (0, 1, 2, 3, 4),
    missing_range: Tuple[int, int] = (1, 3)
) -> Tuple[List[Dict], Dict]:
    """Generate dataset with exact paper specifications and manifest."""
    rng = random.Random(seed)
    tasks = []
    manifest = {
        "total_tasks": num_tasks,
        "seed": seed,
        "grid_mix": grid_mix,
        "alphabet": alphabet,
        "missing_range": missing_range,
        "tasks": []
    }
    
    for k in range(num_tasks):
        # Grid size selection based on mix
        r = rng.random()
        if r < grid_mix[0]:
            grid_size = (3, 3)
        elif r < grid_mix[0] + grid_mix[1]:
            grid_size = (4, 4)
        else:
            grid_size = (5, 5)
        
        # Missing cells selection (mean ~1.7)
        missing = rng.choice(range(missing_range[0], missing_range[1] + 1))
        
        # Generate task
        task_seed = seed + k
        initial, target = generate_task(grid_size, alphabet, missing, task_seed)
        dmax = compute_depth_bound(missing)
        
        task_data = {
            "id": k,
            "initial": initial.tolist(),
            "target": target.tolist(),
            "grid_size": grid_size,
            "missing": missing,
            "dmax": dmax,
            "seed": task_seed
        }
        
        tasks.append(task_data)
        manifest["tasks"].append({
            "id": k,
            "grid_size": grid_size,
            "missing": missing,
            "dmax": dmax,
            "seed": task_seed
        })
    
    return tasks, manifest

def save_manifest(manifest: Dict, path: Union[str, Path]) -> None:
    """Save manifest to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(manifest, f, indent=2)

def load_manifest(path: Union[str, Path]) -> Dict:
    """Load manifest from JSON file."""
    with Path(path).open('r') as f:
        return json.load(f)

def generate_dataset(num_tasks=50, seed=42) -> List[Dict]:
    """Legacy function for backward compatibility."""
    tasks, _ = generate_dataset_spec(num_tasks, seed)
    return tasks
