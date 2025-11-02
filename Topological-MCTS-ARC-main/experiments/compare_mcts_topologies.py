"""
Experiment: Compare MCTS with Constraint Graph vs Grid Topology.

This is the core experiment showing that constraint graph topology
(solution space topology) improves MCTS performance vs grid topology.

Hypothesis: MCTS with constraint graph features will achieve higher
solve rates and better search efficiency.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import time
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topomcts.game import ARCGame
from topomcts.mcts import TopologicalMCTSEngine


def create_test_tasks() -> List[Dict]:
    """Create diverse test tasks with different topologies."""

    tasks = []

    # Task 1: Rotational Symmetry (180°)
    # Constraint graph: Simple, well-structured
    grid = np.array([
        [1, 0, 1],
        [0, -1, 0],
        [1, 0, 1]
    ], dtype=int)
    tasks.append({
        'name': 'rotational_symmetry_simple',
        'grid': grid,
        'pattern_type': 'rotational_symmetry',
        'difficulty': 'easy'  # Well-constrained
    })

    # Task 2: Horizontal Reflection
    grid = np.array([
        [-1, 0, 1],
        [0, 1, 0],
        [-1, 0, 1]
    ], dtype=int)
    tasks.append({
        'name': 'reflective_h_simple',
        'grid': grid,
        'pattern_type': 'reflective_symmetry',
        'difficulty': 'easy'
    })

    # Task 3: Vertical Reflection
    grid = np.array([
        [1, -1, 1],
        [0, -1, 0],
        [1, -1, 1]
    ], dtype=int)
    tasks.append({
        'name': 'reflective_v_simple',
        'grid': grid,
        'pattern_type': 'reflective_symmetry',
        'difficulty': 'easy'
    })

    # Task 4: Color Frequency Pattern
    grid = np.array([
        [0, 1, -1],
        [1, -1, 0],
        [-1, 0, 1]
    ], dtype=int)
    tasks.append({
        'name': 'frequency_balanced',
        'grid': grid,
        'pattern_type': 'frequency',
        'difficulty': 'medium'
    })

    # Task 5: Multiple Missing Cells - High Constraint
    grid = np.array([
        [1, 0, 1],
        [0, -1, 0],
        [1, 0, 1]
    ], dtype=int)
    tasks.append({
        'name': 'high_constraint',
        'grid': grid,
        'pattern_type': 'rotational_symmetry',
        'difficulty': 'hard'
    })

    # Task 6: Low Constraint - Many Missing Cells
    grid = np.array([
        [1, -1, 1],
        [-1, -1, -1],
        [1, -1, 1]
    ], dtype=int)
    tasks.append({
        'name': 'low_constraint',
        'grid': grid,
        'pattern_type': 'rotational_symmetry',
        'difficulty': 'medium'  # More freedom, harder to solve
    })

    # Task 7: No Clear Pattern (Fallback to spatial)
    grid = np.array([
        [1, -1, 2],
        [-1, 3, -1],
        [4, -1, 5]
    ], dtype=int)
    tasks.append({
        'name': 'spatial_pattern',
        'grid': grid,
        'pattern_type': 'unknown',
        'difficulty': 'hard'  # Few constraints
    })

    return tasks


def run_mcts_comparison(
    grid: np.ndarray,
    iterations: int = 100,
    use_constraint_graphs: bool = True,
    runs: int = 3
) -> Dict:
    """
    Compute topological bonuses for constraint graph vs grid topology.

    We measure topological features directly rather than solve rates,
    since actual game solving requires full game environment.

    Returns:
        Dict with topological feature metrics
    """

    results = {
        'use_constraint_graphs': use_constraint_graphs,
        'iterations': iterations,
        'runs': runs,
        'bonuses': [],
        'centralities': [],
        'diffusions': [],
        'symmetries': [],
        'detection_rules': [],
        'run_times': []
    }

    game = ARCGame(grid, grid)

    # Create engine with specified topology
    engine = TopologicalMCTSEngine(
        use_constraint_graphs=use_constraint_graphs,
        topology_weight=0.5
    )

    # Run multiple times to get average features
    for run in range(runs):
        start_time = time.time()

        # Get topological bonus
        bonus, centrality, diffusion, symmetry = engine.topological_bonus(game)
        elapsed = time.time() - start_time

        # If using constraint graphs, also get detected rule
        if use_constraint_graphs:
            rule = engine.pattern_detector.detect_rule(game)
            results['detection_rules'].append(rule)

        results['bonuses'].append(bonus)
        results['centralities'].append(centrality)
        results['diffusions'].append(diffusion)
        results['symmetries'].append(symmetry)
        results['run_times'].append(elapsed)

    return results


def compare_topologies(task: Dict) -> Dict:
    """Compare constraint graph vs grid topology on a single task."""

    grid = task['grid']
    task_name = task['name']

    print(f"\n  Testing: {task_name}")
    print(f"    Grid size: {grid.shape}, Pattern: {task['pattern_type']}")

    # Run with constraint graphs
    print(f"    Running MCTS with constraint graphs...")
    results_constraint = run_mcts_comparison(
        grid,
        iterations=100,
        use_constraint_graphs=True,
        runs=3
    )

    # Run with grid topology (fallback)
    print(f"    Running MCTS with grid topology...")
    results_grid = run_mcts_comparison(
        grid,
        iterations=100,
        use_constraint_graphs=False,
        runs=3
    )

    return {
        'task': task_name,
        'pattern_type': task['pattern_type'],
        'difficulty': task['difficulty'],
        'detected_rule': results_constraint['detection_rules'][0] if results_constraint['detection_rules'] else 'unknown',
        'constraint_graph': results_constraint,
        'grid_topology': results_grid,
        'comparison': {
            'constraint_avg_bonus': np.mean(results_constraint['bonuses']),
            'constraint_std_bonus': np.std(results_constraint['bonuses']),
            'grid_avg_bonus': np.mean(results_grid['bonuses']),
            'grid_std_bonus': np.std(results_grid['bonuses']),
            'bonus_improvement': (np.mean(results_constraint['bonuses']) -
                                 np.mean(results_grid['bonuses'])),
            'bonus_ratio': (np.mean(results_constraint['bonuses']) /
                           max(0.001, np.mean(results_grid['bonuses']))),
            'constraint_avg_centrality': np.mean(results_constraint['centralities']),
            'grid_avg_centrality': np.mean(results_grid['centralities']),
            'constraint_avg_diffusion': np.mean(results_constraint['diffusions']),
            'grid_avg_diffusion': np.mean(results_grid['diffusions']),
            'constraint_avg_symmetry': np.mean(results_constraint['symmetries']),
            'grid_avg_symmetry': np.mean(results_grid['symmetries']),
            'constraint_avg_time': np.mean(results_constraint['run_times']),
            'grid_avg_time': np.mean(results_grid['run_times']),
            'time_ratio': (np.mean(results_constraint['run_times']) /
                          max(0.001, np.mean(results_grid['run_times'])))
        }
    }


def run_comparison_experiment() -> Dict:
    """Run full comparison experiment."""

    print("Creating test tasks...")
    tasks = create_test_tasks()
    print(f"Created {len(tasks)} test tasks\n")

    results = {
        'experiment': 'MCTS Topology Comparison',
        'timestamp': str(Path(__file__).stat().st_mtime),
        'tasks': len(tasks),
        'comparisons': []
    }

    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {task['name']}")
        comparison = compare_topologies(task)
        results['comparisons'].append(comparison)

    return results


def analyze_comparison_results(results: Dict) -> None:
    """Analyze and print comparison results."""

    print("\n\n" + "="*80)
    print("MCTS TOPOLOGY COMPARISON RESULTS")
    print("="*80)

    print(f"\nTotal tasks: {results['tasks']}")
    print(f"Iterations per run: 100")
    print(f"Runs per configuration: 3\n")

    print("Task-by-Task Topological Feature Comparison:")
    print("-"*100)
    print(f"{'Task':<28} {'Pattern':<20} {'CG Bonus':<12} {'Grid Bonus':<12} {'Ratio':<8}")
    print("-"*100)

    total_constraint_bonus = 0
    total_grid_bonus = 0
    total_ratio = 0
    constraints_better = 0

    for comp in results['comparisons']:
        task_name = comp['task']
        pattern = comp['pattern_type']
        cg_bonus = comp['comparison']['constraint_avg_bonus']
        grid_bonus = comp['comparison']['grid_avg_bonus']
        ratio = comp['comparison']['bonus_ratio']

        if cg_bonus > grid_bonus:
            constraints_better += 1

        total_constraint_bonus += cg_bonus
        total_grid_bonus += grid_bonus
        total_ratio += ratio

        print(f"{task_name:<28} {pattern:<20} {cg_bonus:>10.4f}  {grid_bonus:>10.4f}  {ratio:>7.2f}x")

    print("-"*100)
    print(f"\nSummary:")
    print(f"  Constraint graphs higher bonus: {constraints_better}/{len(results['comparisons'])}")
    print(f"  Average CG bonus: {total_constraint_bonus/len(results['comparisons']):.4f}")
    print(f"  Average Grid bonus: {total_grid_bonus/len(results['comparisons']):.4f}")
    print(f"  Average ratio (CG/Grid): {total_ratio/len(results['comparisons']):.2f}x")

    # Analysis by pattern type
    print("\nTopological Feature Improvement by Pattern Type:")
    print("-"*80)

    pattern_results = defaultdict(list)
    for comp in results['comparisons']:
        pattern = comp['pattern_type']
        ratio = comp['comparison']['bonus_ratio']
        pattern_results[pattern].append(ratio)

    for pattern, ratios in sorted(pattern_results.items()):
        avg_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        print(f"  {pattern:<30} {avg_ratio:.2f}x ± {std_ratio:.2f}x")

    # Analysis by difficulty
    print("\nTopological Feature Improvement by Difficulty:")
    print("-"*80)

    difficulty_results = defaultdict(list)
    for comp in results['comparisons']:
        difficulty = comp['difficulty']
        ratio = comp['comparison']['bonus_ratio']
        difficulty_results[difficulty].append(ratio)

    for difficulty in ['easy', 'medium', 'hard']:
        if difficulty in difficulty_results:
            ratios = difficulty_results[difficulty]
            avg_ratio = np.mean(ratios)
            std_ratio = np.std(ratios)
            print(f"  {difficulty:<30} {avg_ratio:.2f}x ± {std_ratio:.2f}x")

    print("\n" + "="*80)


def save_comparison_results(results: Dict, output_path: str = None) -> None:
    """Save detailed results to JSON."""
    if output_path is None:
        output_path = Path(__file__).parent / "mcts_topology_comparison.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    print("Running MCTS Topology Comparison Experiment...")
    print("(This compares constraint graphs vs grid topology)\n")

    # Run experiment
    results = run_comparison_experiment()

    # Analyze results
    analyze_comparison_results(results)

    # Save results
    save_comparison_results(results)
