#!/usr/bin/env python3
"""
Experiment: Test Topological MCTS on real ARC-1 tasks.

Loads selected middle-difficulty ARC-1 tasks and evaluates:
1. Pattern detection accuracy on real tasks
2. MCTS solving success rate with and without topological guidance
3. Sample efficiency (number of rollouts to solve)
4. Runtime overhead

Outputs metrics to CSV and JSON for analysis.
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from topomcts.game import ARCGame
from topomcts.pattern_detector import PatternDetector
from topomcts.constraint_graph import ConstraintGraph
from topomcts.mcts import TopologicalMCTSEngine
from topomcts.baselines import StandardMCTSEngine


@dataclass
class TaskResult:
    """Results for a single ARC task."""
    task_id: str
    train_size: int  # Number of training examples
    test_input_shape: Tuple[int, int]
    test_missing_cells: int

    # Pattern detection
    patterns_detected: int
    pattern_types: List[str]

    # Baseline MCTS
    baseline_success: bool
    baseline_rollouts: int
    baseline_time_sec: float

    # Topological MCTS
    topo_success: bool
    topo_rollouts: int
    topo_time_sec: float

    # Comparison
    efficiency_gain: float  # rollouts_baseline / rollouts_topo (>1 means topo is better)


def load_arc_task(task_id: str, arc_data_dir: Path) -> Optional[dict]:
    """Load a single ARC task JSON."""
    task_file = arc_data_dir / f"{task_id}.json"
    if not task_file.exists():
        return None
    with open(task_file, 'r') as f:
        return json.load(f)


def convert_arc_to_game(task: dict) -> Tuple[Optional[ARCGame], Dict]:
    """
    Convert ARC task to ARCGame format.

    ARC format: list of (input, output) pairs
    Game format: single grid with missing cells (-1), target grid

    Strategy: Use first training example as pattern reference,
    use test input as grid to complete.
    """
    try:
        if not task.get('train') or not task.get('test'):
            return None, {}

        # Use test input as the grid to solve
        test_pair = task['test'][0]
        test_input = np.array(test_pair['input'])
        test_output = np.array(test_pair['output'])

        # Check output exists and shapes are compatible
        if 'output' not in test_pair:
            return None, {}

        # For simplicity: assume test output shape is correct
        # Create initial state by masking some cells as missing
        initial = test_input.copy()

        # Get alphabet from training data
        all_colors = set()
        for pair in task['train']:
            all_colors.update(np.unique(pair['input']).tolist())
            all_colors.update(np.unique(pair['output']).tolist())
        for pair in task['test']:
            all_colors.update(np.unique(pair['input']).tolist())
            if 'output' in pair:
                all_colors.update(np.unique(pair['output']).tolist())

        alphabet = sorted(list(all_colors))

        game = ARCGame(initial, test_output, alphabet)

        return game, {
            'train_pairs': len(task['train']),
            'test_shape': test_output.shape,
            'missing_cells': len(game.missing_positions()),
            'alphabet_size': len(alphabet)
        }

    except Exception as e:
        return None, {}


def detect_patterns_in_task(task: dict) -> Tuple[int, List[str]]:
    """Detect patterns from training examples."""
    try:
        detector = PatternDetector()
        patterns = []
        pattern_types_set = set()

        for pair in task.get('train', []):
            inp = np.array(pair['input'])
            out = np.array(pair['output'])

            detected = detector.detect(inp, out)
            if detected:
                patterns.append(detected)
                pattern_types_set.add(detected.pattern_type)

        return len(patterns), list(pattern_types_set)
    except:
        return 0, []


def run_mcts_baseline(
    game: ARCGame,
    max_rollouts: int = 100,
    timeout_sec: float = 30.0
) -> Tuple[bool, int, float]:
    """Run baseline MCTS without topological guidance."""
    try:
        engine = StandardMCTSEngine(ucb_c=1.414, rollout_depth=10)
        start = time.time()

        rollouts = 0
        success = False
        start_time = time.time()

        while rollouts < max_rollouts and (time.time() - start_time) < timeout_sec:
            action, value, _ = engine.run(game, iterations=1)
            if value >= 1.0:  # Found solution
                success = True
                break
            rollouts += 1

        elapsed = time.time() - start
        return success, rollouts, elapsed
    except Exception as e:
        print(f"  Baseline MCTS failed: {e}")
        return False, max_rollouts, 0.0


def run_mcts_topological(
    game: ARCGame,
    task: dict,
    max_rollouts: int = 100,
    timeout_sec: float = 30.0
) -> Tuple[bool, int, float]:
    """Run MCTS with topological guidance."""
    try:
        engine = TopologicalMCTSEngine(
            ucb_c=1.414,
            rollout_depth=10,
            topology_weight=0.5,
            use_constraint_graphs=True
        )
        start = time.time()

        rollouts = 0
        success = False
        start_time = time.time()

        while rollouts < max_rollouts and (time.time() - start_time) < timeout_sec:
            action, value, _ = engine.run(game, iterations=1)
            if value >= 1.0:  # Found solution
                success = True
                break
            rollouts += 1

        elapsed = time.time() - start
        return success, rollouts, elapsed
    except Exception as e:
        print(f"  Topological MCTS failed: {e}")
        return False, max_rollouts, 0.0


def run_arc1_experiments(
    selected_task_file: Path,
    arc_data_dir: Path,
    output_dir: Path,
    max_rollouts: int = 100,
    timeout_sec: float = 30.0
) -> List[TaskResult]:
    """Run experiments on selected ARC-1 tasks."""

    os.makedirs(output_dir, exist_ok=True)

    # Load selected task IDs
    with open(selected_task_file, 'r') as f:
        task_ids = json.load(f)

    print(f"\nRunning experiments on {len(task_ids)} ARC-1 tasks")
    print(f"Max rollouts per task: {max_rollouts}")
    print(f"Timeout per task: {timeout_sec}s")
    print("=" * 80)

    results = []

    for idx, task_id in enumerate(task_ids, 1):
        print(f"\n[{idx}/{len(task_ids)}] Task: {task_id}")

        # Load task
        task_data = load_arc_task(task_id, arc_data_dir)
        if not task_data:
            print(f"  ✗ Failed to load task")
            continue

        # Convert to game
        game, game_info = convert_arc_to_game(task_data)
        if not game:
            print(f"  ✗ Failed to convert to game format")
            continue

        print(f"    Grid shape: {game_info['test_shape']}")
        print(f"    Missing cells: {game_info['missing_cells']}")
        print(f"    Alphabet size: {game_info['alphabet_size']}")

        # Detect patterns
        num_patterns, pattern_types = detect_patterns_in_task(task_data)
        print(f"    Patterns detected: {num_patterns} ({', '.join(pattern_types) if pattern_types else 'none'})")

        # Run baseline MCTS
        print(f"    Running baseline MCTS...", end="", flush=True)
        baseline_success, baseline_rollouts, baseline_time = run_mcts_baseline(
            game, max_rollouts, timeout_sec
        )
        print(f" {'✓' if baseline_success else '✗'} ({baseline_rollouts} rollouts, {baseline_time:.2f}s)")

        # Run topological MCTS
        print(f"    Running topological MCTS...", end="", flush=True)
        topo_success, topo_rollouts, topo_time = run_mcts_topological(
            game, task_data, max_rollouts, timeout_sec
        )
        print(f" {'✓' if topo_success else '✗'} ({topo_rollouts} rollouts, {topo_time:.2f}s)")

        # Calculate efficiency gain
        efficiency_gain = baseline_rollouts / topo_rollouts if topo_rollouts > 0 else 0

        result = TaskResult(
            task_id=task_id,
            train_size=game_info['train_pairs'],
            test_input_shape=game_info['test_shape'],
            test_missing_cells=game_info['missing_cells'],
            patterns_detected=num_patterns,
            pattern_types=pattern_types,
            baseline_success=baseline_success,
            baseline_rollouts=baseline_rollouts,
            baseline_time_sec=baseline_time,
            topo_success=topo_success,
            topo_rollouts=topo_rollouts,
            topo_time_sec=topo_time,
            efficiency_gain=efficiency_gain
        )

        results.append(result)

    return results


def save_results(results: List[TaskResult], output_dir: Path) -> None:
    """Save results to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Save as JSON
    json_file = output_dir / "arc1_results.json"
    with open(json_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {json_file}")

    # Save as CSV
    csv_file = output_dir / "arc1_results.csv"
    with open(csv_file, 'w') as f:
        # Header
        if results:
            header = ','.join(asdict(results[0]).keys())
            f.write(header + '\n')

            # Rows
            for r in results:
                row_data = asdict(r)
                # Convert lists to quoted strings
                row_values = []
                for v in row_data.values():
                    if isinstance(v, list):
                        row_values.append(f'"{"|".join(map(str, v))}"')
                    else:
                        row_values.append(str(v))
                f.write(','.join(row_values) + '\n')

    print(f"Results saved to {csv_file}")

    # Print summary statistics
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)

        baseline_success = sum(1 for r in results if r.baseline_success)
        topo_success = sum(1 for r in results if r.topo_success)

        print(f"Tasks evaluated: {len(results)}")
        print(f"Baseline success rate: {baseline_success}/{len(results)} ({100*baseline_success/len(results):.1f}%)")
        print(f"Topological success rate: {topo_success}/{len(results)} ({100*topo_success/len(results):.1f}%)")

        # Efficiency for successful tasks
        successful_results = [r for r in results if r.topo_success and r.baseline_success]
        if successful_results:
            gains = [r.efficiency_gain for r in successful_results]
            print(f"\nEfficiency gains (on successful solves):")
            print(f"  Mean: {np.mean(gains):.2f}x")
            print(f"  Median: {np.median(gains):.2f}x")
            print(f"  Min: {np.min(gains):.2f}x")
            print(f"  Max: {np.max(gains):.2f}x")

        # Pattern detection stats
        avg_patterns = np.mean([r.patterns_detected for r in results])
        print(f"\nPattern detection:")
        print(f"  Average patterns per task: {avg_patterns:.1f}")


if __name__ == '__main__':
    # Paths
    script_dir = Path(__file__).parent
    arc_data_dir = Path(__file__).parent.parent.parent.parent / "ARC-AGI" / "data" / "training"
    selected_task_file = script_dir / "selected_task_ids.json"
    output_dir = script_dir / "results"

    # Verify paths
    if not selected_task_file.exists():
        print(f"Error: Selected tasks file not found at {selected_task_file}")
        print("Run select_arc_tasks.py first.")
        sys.exit(1)

    if not arc_data_dir.exists():
        print(f"Error: ARC-AGI data directory not found at {arc_data_dir}")
        sys.exit(1)

    # Run experiments
    results = run_arc1_experiments(
        selected_task_file,
        arc_data_dir,
        output_dir,
        max_rollouts=100,
        timeout_sec=30.0
    )

    # Save results
    save_results(results, output_dir)
