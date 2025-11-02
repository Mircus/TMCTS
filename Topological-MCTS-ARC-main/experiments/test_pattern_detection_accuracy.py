"""
Experiment: Measure pattern detection accuracy on ARC tasks.

This script loads real ARC training tasks and tests pattern detection:
- What fraction of tasks have detectable patterns?
- Which pattern types are most common?
- How accurate is detection across different task families?
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topomcts.game import ARCGame
from topomcts.pattern_detector import PatternDetector


def load_arc_tasks(max_tasks: int = None) -> List[Dict]:
    """Load ARC training tasks from JSON."""
    arc_path = Path(__file__).parent.parent / "data" / "ARC" / "training"

    if not arc_path.exists():
        print(f"Warning: ARC data not found at {arc_path}")
        print("Creating synthetic test tasks instead...")
        return generate_synthetic_tasks(max_tasks or 10)

    tasks = []
    for task_file in sorted(arc_path.glob("*.json"))[:max_tasks]:
        try:
            with open(task_file) as f:
                task = json.load(f)
                tasks.append({
                    'name': task_file.stem,
                    'task': task
                })
        except Exception as e:
            print(f"Error loading {task_file}: {e}")

    return tasks


def generate_synthetic_tasks(count: int = 10) -> List[Dict]:
    """Generate synthetic test tasks with known patterns."""
    tasks = []

    # Rotational symmetry tasks
    for i in range(count // 4):
        grid = np.array([
            [1, 0, 1],
            [0, 2, 0],
            [1, 0, 1]
        ])
        tasks.append({
            'name': f'synthetic_rotational_180_{i}',
            'task': {'train': [{'input': grid.tolist(), 'output': grid.tolist()}]}
        })

    # Reflective symmetry tasks
    for i in range(count // 4):
        grid = np.array([
            [1, 2, 1],
            [3, 4, 3],
            [5, 6, 5]
        ])
        tasks.append({
            'name': f'synthetic_reflective_v_{i}',
            'task': {'train': [{'input': grid.tolist(), 'output': grid.tolist()}]}
        })

    # Color frequency tasks
    for i in range(count // 4):
        grid = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2]
        ])
        tasks.append({
            'name': f'synthetic_frequency_{i}',
            'task': {'train': [{'input': grid.tolist(), 'output': grid.tolist()}]}
        })

    # Progression tasks
    for i in range(count // 4):
        grid = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        tasks.append({
            'name': f'synthetic_progression_{i}',
            'task': {'train': [{'input': grid.tolist(), 'output': grid.tolist()}]}
        })

    return tasks


def extract_first_example_grid(task: Dict) -> np.ndarray:
    """Extract input grid from first training example."""
    if 'train' in task and len(task['train']) > 0:
        return np.array(task['train'][0]['input'], dtype=int)
    elif 'test' in task and len(task['test']) > 0:
        return np.array(task['test'][0]['input'], dtype=int)
    else:
        return None


def test_pattern_detection() -> Dict:
    """Test pattern detection accuracy on ARC tasks."""

    print("Loading ARC tasks...")
    tasks = load_arc_tasks(max_tasks=50)  # Start with 50 tasks
    print(f"Loaded {len(tasks)} tasks")

    detector = PatternDetector()
    results = {
        'total_tasks': len(tasks),
        'pattern_counts': defaultdict(int),
        'patterns_detected': [],
        'patterns_failed': [],
        'detection_errors': 0,
        'spatial_pattern_tasks': []
    }

    for task_info in tasks:
        task_name = task_info['name']
        task = task_info['task']

        try:
            # Extract first training example
            grid = extract_first_example_grid(task)

            if grid is None or grid.size == 0:
                results['detection_errors'] += 1
                continue

            # Create game state (using same grid for input/output)
            # In real scenario, we'd use just the input
            game = ARCGame(grid, grid)

            # Detect pattern
            rule = detector.detect_rule(game)
            results['pattern_counts'][rule] += 1
            results['patterns_detected'].append({
                'task': task_name,
                'pattern': rule,
                'grid_size': grid.shape
            })

            if rule == 'spatial_pattern':
                results['spatial_pattern_tasks'].append(task_name)

        except Exception as e:
            results['detection_errors'] += 1
            results['patterns_failed'].append({
                'task': task_name,
                'error': str(e)
            })
            print(f"  Error on {task_name}: {e}")

    return results


def analyze_results(results: Dict) -> None:
    """Analyze and print detection results."""

    print("\n" + "="*60)
    print("PATTERN DETECTION RESULTS")
    print("="*60)

    total = results['total_tasks']
    detected = len(results['patterns_detected'])
    errors = results['detection_errors']

    print(f"\nTotal tasks analyzed: {total}")
    print(f"Successfully detected: {detected}")
    print(f"Detection errors: {errors}")
    print(f"Detection rate: {100*detected/max(1, total):.1f}%")

    print("\nPattern Distribution:")
    print("-" * 60)

    pattern_counts = dict(results['pattern_counts'])
    total_detected = sum(pattern_counts.values())

    for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(1, total_detected)
        print(f"  {pattern:30s}: {count:3d} ({pct:5.1f}%)")

    print("\nPattern Type Analysis:")
    print("-" * 60)

    symmetry_count = sum(c for p, c in pattern_counts.items() if 'symmetry' in p)
    frequency_count = pattern_counts.get('color_frequency', 0)
    progression_count = pattern_counts.get('arithmetic_progression', 0)
    spatial_count = pattern_counts.get('spatial_pattern', 0)

    print(f"  Symmetry patterns:      {symmetry_count:3d} ({100*symmetry_count/max(1,total_detected):5.1f}%)")
    print(f"  Frequency pattern:      {frequency_count:3d} ({100*frequency_count/max(1,total_detected):5.1f}%)")
    print(f"  Progression pattern:    {progression_count:3d} ({100*progression_count/max(1,total_detected):5.1f}%)")
    print(f"  Spatial pattern (unknown): {spatial_count:3d} ({100*spatial_count/max(1,total_detected):5.1f}%)")

    # Print sample detections
    print("\nSample Detections (first 10):")
    print("-" * 60)
    for i, detection in enumerate(results['patterns_detected'][:10]):
        print(f"  {i+1}. {detection['task']}")
        print(f"     Pattern: {detection['pattern']}, Grid: {detection['grid_size']}")

    # Print failures if any
    if results['patterns_failed']:
        print("\nDetection Failures:")
        print("-" * 60)
        for failure in results['patterns_failed'][:5]:
            print(f"  {failure['task']}: {failure['error']}")
        if len(results['patterns_failed']) > 5:
            print(f"  ... and {len(results['patterns_failed']) - 5} more")

    print("\n" + "="*60)


def save_results(results: Dict, output_path: str = None) -> None:
    """Save results to JSON file."""
    if output_path is None:
        output_path = Path(__file__).parent / "pattern_detection_results.json"

    # Convert defaultdict to dict for JSON serialization
    results['pattern_counts'] = dict(results['pattern_counts'])
    results['spatial_pattern_tasks'] = results['spatial_pattern_tasks'][:10]  # Limit output

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    print("Running pattern detection accuracy experiment...")
    print()

    # Run experiment
    results = test_pattern_detection()

    # Analyze and display results
    analyze_results(results)

    # Save results
    save_results(results)
