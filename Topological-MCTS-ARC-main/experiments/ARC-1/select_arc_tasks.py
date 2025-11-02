#!/usr/bin/env python3
"""
Select middle-difficulty ARC-1 tasks based on complexity metrics.

Complexity is measured by:
- Grid size (area)
- Number of unique colors in train/test
- Number of missing cells in test
- Variance in grid dimensions

Middle-difficulty: tasks with moderate complexity, likely solvable with pattern detection
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

def compute_task_complexity(task: dict) -> float:
    """
    Compute complexity score for a task.
    Higher score = more complex.

    Factors:
    - Grid area (larger = more complex)
    - Color diversity (more unique colors = more complex)
    - Test difficulty (infer missing cells from shape mismatch)
    """
    scores = []

    # Analyze training examples
    for pair in task.get('train', []):
        inp = np.array(pair['input'])
        out = np.array(pair['output'])

        # Grid size factor
        area = inp.size
        scores.append(area / 1000.0)  # Normalize by 1000

        # Color diversity in input
        unique_colors = len(np.unique(inp))
        scores.append(unique_colors / 10.0)  # Normalize by 10

        # Shape change (input != output shape = transformation)
        shape_change = float(inp.shape != out.shape)
        scores.append(shape_change)

    # Analyze test examples
    for pair in task.get('test', []):
        inp = np.array(pair['input'])

        # Test input size
        area = inp.size
        scores.append(area / 1000.0)

        # Color diversity in test input
        unique_colors = len(np.unique(inp))
        scores.append(unique_colors / 10.0)

    # Average complexity
    return np.mean(scores) if scores else 0.0


def select_middle_difficulty_tasks(
    data_dir: str,
    num_tasks: int = 20,
    percentile_low: float = 30.0,
    percentile_high: float = 70.0
) -> List[Tuple[str, dict]]:
    """
    Select tasks from middle difficulty range.

    Args:
        data_dir: Path to training data directory
        num_tasks: Target number of tasks to select
        percentile_low: Lower percentile for difficulty range
        percentile_high: Upper percentile for difficulty range

    Returns:
        List of (task_id, task_dict) tuples
    """
    tasks = {}
    complexities = {}

    # Load all tasks and compute complexity
    for fname in os.listdir(data_dir):
        if not fname.endswith('.json'):
            continue

        task_id = fname.replace('.json', '')
        with open(os.path.join(data_dir, fname), 'r') as f:
            task = json.load(f)

        complexity = compute_task_complexity(task)
        tasks[task_id] = task
        complexities[task_id] = complexity

    # Find middle difficulty range
    complexity_values = np.array(list(complexities.values()))
    low_threshold = np.percentile(complexity_values, percentile_low)
    high_threshold = np.percentile(complexity_values, percentile_high)

    # Select tasks in middle range
    middle_tasks = [
        (tid, tasks[tid])
        for tid, comp in complexities.items()
        if low_threshold <= comp <= high_threshold
    ]

    # Sort by complexity and select evenly distributed sample
    middle_tasks.sort(key=lambda x: complexities[x[0]])

    # Sample evenly across the range
    if len(middle_tasks) > num_tasks:
        indices = np.linspace(0, len(middle_tasks) - 1, num_tasks, dtype=int)
        middle_tasks = [middle_tasks[i] for i in indices]

    return middle_tasks


def save_selected_tasks(
    selected_tasks: List[Tuple[str, dict]],
    output_dir: str
) -> None:
    """Save selected tasks to a JSON file for easy reference."""
    os.makedirs(output_dir, exist_ok=True)

    # Save task IDs list
    task_ids = [tid for tid, _ in selected_tasks]
    with open(os.path.join(output_dir, 'selected_task_ids.json'), 'w') as f:
        json.dump(task_ids, f, indent=2)

    # Save full tasks
    tasks_dict = {tid: task for tid, task in selected_tasks}
    with open(os.path.join(output_dir, 'selected_tasks.json'), 'w') as f:
        json.dump(tasks_dict, f, indent=2)

    # Print summary
    print(f"Selected {len(selected_tasks)} middle-difficulty tasks")
    print(f"\nTask IDs:")
    for i, tid in enumerate(task_ids, 1):
        print(f"  {i:2d}. {tid}")


if __name__ == '__main__':
    import sys

    # Paths
    arc_data_dir = Path(__file__).parent.parent.parent.parent / "ARC-AGI" / "data" / "training"
    output_dir = Path(__file__).parent

    if not arc_data_dir.exists():
        print(f"Error: ARC-AGI data directory not found at {arc_data_dir}")
        print("Please ensure ARC-AGI is cloned to the parent directory.")
        sys.exit(1)

    # Select middle-difficulty tasks
    print(f"Loading tasks from {arc_data_dir}...")
    selected = select_middle_difficulty_tasks(
        str(arc_data_dir),
        num_tasks=20,
        percentile_low=35.0,
        percentile_high=65.0
    )

    # Save selected tasks
    save_selected_tasks(selected, str(output_dir))
    print(f"\nSelected tasks saved to {output_dir}")
