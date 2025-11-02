#!/usr/bin/env python3
"""
Real ARC-1 Task Evaluation using ARCTransformationGame.

Tests both baseline and topological MCTS on 20 real ARC-1 tasks.
Measures success rate, rollout efficiency, and topological advantage.
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

from topomcts.arc_transformation_game import ARCTransformationGame
from topomcts.pattern_detector import PatternDetector
from topomcts.mcts import TopologicalMCTSEngine
from topomcts.baselines import StandardMCTSEngine


@dataclass
class TaskResult:
    """Results for a single real ARC task."""
    task_id: str
    test_shape: Tuple[int, int]
    fillable_cells: int
    alphabet_size: int

    # Detected pattern
    pattern_rule: str

    # Baseline MCTS results
    baseline_solved: bool
    baseline_rollouts: int
    baseline_quality: float
    baseline_time_sec: float

    # Topological MCTS results
    topo_solved: bool
    topo_rollouts: int
    topo_quality: float
    topo_time_sec: float

    # Comparison metrics
    efficiency_gain: float  # rollouts_baseline / rollouts_topo (>1 means topo better)
    quality_gain: float  # topo_quality - baseline_quality


def load_arc_task(task_id: str, arc_data_dir: Path) -> Optional[dict]:
    """Load a single ARC task JSON."""
    task_file = arc_data_dir / f"{task_id}.json"
    if not task_file.exists():
        return None
    with open(task_file, 'r') as f:
        return json.load(f)


def detect_pattern(game: ARCTransformationGame) -> str:
    """Detect pattern from training examples."""
    try:
        detector = PatternDetector()
        if game.training_examples:
            # Use first training example to detect pattern
            first_example = game.training_examples[0]
            inp = np.array(first_example['input'])
            out = np.array(first_example['output'])

            # Create a temporary game-like object for pattern detection
            class TempGame:
                def __init__(self, grid):
                    self.initial = grid

            temp = TempGame(inp)
            rule = detector.detect_rule(temp)
            return rule
        return "unknown"
    except:
        return "unknown"


def run_mcts_baseline(
    game: ARCTransformationGame,
    max_rollouts: int = 50,
    iterations_per_rollout: int = 20,
    timeout_sec: float = 30.0
) -> Tuple[bool, int, float, float]:
    """
    Run baseline MCTS without topological guidance.

    Returns: (solved, rollouts_used, quality, elapsed_time)
    """
    try:
        engine = StandardMCTSEngine(ucb_c=1.414, rollout_depth=15)
        start = time.time()
        best_quality = 0.0
        rollouts = 0

        for rollout in range(max_rollouts):
            if time.time() - start > timeout_sec:
                break

            # Run one iteration of MCTS
            action, value, metadata = engine.run(game, iterations=iterations_per_rollout)

            # Check current best quality
            current_quality = game.quality()
            if current_quality > best_quality:
                best_quality = current_quality

            # Check if solved
            if current_quality >= 1.0:
                elapsed = time.time() - start
                return True, rollouts, current_quality, elapsed

            rollouts += 1

        elapsed = time.time() - start
        return False, rollouts, best_quality, elapsed

    except Exception as e:
        return False, max_rollouts, 0.0, 0.0


def run_mcts_topological(
    game: ARCTransformationGame,
    max_rollouts: int = 50,
    iterations_per_rollout: int = 20,
    timeout_sec: float = 30.0
) -> Tuple[bool, int, float, float]:
    """
    Run MCTS with topological guidance.

    Returns: (solved, rollouts_used, quality, elapsed_time)
    """
    try:
        engine = TopologicalMCTSEngine(
            ucb_c=1.414,
            rollout_depth=15,
            topology_weight=0.5,
            use_constraint_graphs=True
        )
        start = time.time()
        best_quality = 0.0
        rollouts = 0

        for rollout in range(max_rollouts):
            if time.time() - start > timeout_sec:
                break

            # Run one iteration of MCTS
            try:
                action, value, metadata = engine.run(game, iterations=iterations_per_rollout)
            except Exception:
                # If constraint graph fails, fall back gracefully
                action, value, metadata = None, 0.0, {}

            # Check current best quality
            current_quality = game.quality()
            if current_quality > best_quality:
                best_quality = current_quality

            # Check if solved
            if current_quality >= 1.0:
                elapsed = time.time() - start
                return True, rollouts, current_quality, elapsed

            rollouts += 1

        elapsed = time.time() - start
        return False, rollouts, best_quality, elapsed

    except Exception as e:
        return False, max_rollouts, 0.0, 0.0


def run_real_arc_experiments(
    selected_task_file: Path,
    arc_data_dir: Path,
    output_dir: Path,
    max_rollouts: int = 50,
    iterations_per_rollout: int = 20,
    timeout_sec: float = 30.0
) -> List[TaskResult]:
    """Run experiments on real ARC-1 tasks."""

    os.makedirs(output_dir, exist_ok=True)

    # Load selected task IDs
    with open(selected_task_file, 'r') as f:
        task_ids = json.load(f)

    print(f"\n{'='*80}")
    print(f"REAL ARC-1 TASK EVALUATION")
    print(f"{'='*80}")
    print(f"Tasks: {len(task_ids)}")
    print(f"Max rollouts per task: {max_rollouts}")
    print(f"Iterations per rollout: {iterations_per_rollout}")
    print(f"Timeout per task: {timeout_sec}s")
    print(f"{'='*80}\n")

    results = []

    for idx, task_id in enumerate(task_ids, 1):
        print(f"[{idx:2d}/{len(task_ids)}] Task: {task_id}", end=" ... ", flush=True)

        # Load task
        task_data = load_arc_task(task_id, arc_data_dir)
        if not task_data:
            print("SKIP (load failed)")
            continue

        # Create transformation game
        game = ARCTransformationGame.from_arc_task(task_data, verbose=False)
        if not game:
            print("SKIP (invalid game)")
            continue

        # Detect pattern
        pattern = detect_pattern(game)

        # Get task info
        fillable = len(game.fillable_positions)
        alphabet = len(game.alphabet)

        # Run baseline MCTS
        baseline_solved, baseline_rolls, baseline_quality, baseline_time = run_mcts_baseline(
            game, max_rollouts, iterations_per_rollout, timeout_sec
        )

        # Reset game for topological MCTS
        game = ARCTransformationGame.from_arc_task(task_data, verbose=False)

        # Run topological MCTS
        topo_solved, topo_rolls, topo_quality, topo_time = run_mcts_topological(
            game, max_rollouts, iterations_per_rollout, timeout_sec
        )

        # Calculate gains
        efficiency_gain = baseline_rolls / topo_rolls if topo_rolls > 0 else 0.0
        quality_gain = topo_quality - baseline_quality

        # Status
        status = ""
        if topo_solved:
            status = "SOLVED"
        elif topo_quality >= 0.8:
            status = f"{topo_quality*100:.0f}%"
        else:
            status = f"{topo_quality*100:.0f}%"

        print(status)

        result = TaskResult(
            task_id=task_id,
            test_shape=game.ground_truth.shape,
            fillable_cells=fillable,
            alphabet_size=alphabet,
            pattern_rule=pattern,
            baseline_solved=baseline_solved,
            baseline_rollouts=baseline_rolls,
            baseline_quality=baseline_quality,
            baseline_time_sec=baseline_time,
            topo_solved=topo_solved,
            topo_rollouts=topo_rolls,
            topo_quality=topo_quality,
            topo_time_sec=topo_time,
            efficiency_gain=efficiency_gain,
            quality_gain=quality_gain
        )

        results.append(result)

        # Save partial results after each task
        if idx % 5 == 0 or idx == len(task_ids):
            _save_partial_results(results, output_dir)

    return results


def _save_partial_results(results: List[TaskResult], output_dir: Path) -> None:
    """Save partial results during experiment (internal helper)."""
    if not results:
        return
    os.makedirs(output_dir, exist_ok=True)
    json_file = output_dir / "real_arc_results.json"
    with open(json_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)


def save_results(results: List[TaskResult], output_dir: Path) -> None:
    """Save results to CSV and JSON with summary statistics."""
    os.makedirs(output_dir, exist_ok=True)

    # Save as JSON
    json_file = output_dir / "real_arc_results.json"
    with open(json_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {json_file}")

    # Save as CSV
    csv_file = output_dir / "real_arc_results.csv"
    with open(csv_file, 'w') as f:
        if results:
            # Header
            header = list(asdict(results[0]).keys())
            f.write(','.join(header) + '\n')

            # Rows
            for r in results:
                row = asdict(r)
                values = [str(v) for v in row.values()]
                f.write(','.join(values) + '\n')

    print(f"Results saved to {csv_file}")

    # Print summary statistics
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)

        baseline_solved = sum(1 for r in results if r.baseline_solved)
        topo_solved = sum(1 for r in results if r.topo_solved)

        print(f"\nTasks evaluated: {len(results)}")
        print(f"\nSuccess Rate:")
        print(f"  Baseline (exact match): {baseline_solved}/{len(results)} ({100*baseline_solved/len(results):.1f}%)")
        print(f"  Topological (exact match): {topo_solved}/{len(results)} ({100*topo_solved/len(results):.1f}%)")

        # Quality statistics
        baseline_qualities = [r.baseline_quality for r in results]
        topo_qualities = [r.topo_quality for r in results]

        print(f"\nSolution Quality:")
        print(f"  Baseline:   mean={np.mean(baseline_qualities):.3f}, median={np.median(baseline_qualities):.3f}")
        print(f"  Topological: mean={np.mean(topo_qualities):.3f}, median={np.median(topo_qualities):.3f}")

        quality_gains = [r.quality_gain for r in results]
        print(f"  Quality gain (topo - baseline): mean={np.mean(quality_gains):+.3f}, median={np.median(quality_gains):+.3f}")

        # Efficiency statistics (only where topo solved)
        solved_results = [r for r in results if r.topo_solved and r.baseline_solved]
        if solved_results:
            gains = [r.efficiency_gain for r in solved_results]
            print(f"\nRollout Efficiency (on solved tasks):")
            print(f"  Mean: {np.mean(gains):.2f}x")
            print(f"  Median: {np.median(gains):.2f}x")
            print(f"  Min: {np.min(gains):.2f}x")
            print(f"  Max: {np.max(gains):.2f}x")

        # Pattern distribution
        patterns = {}
        for r in results:
            patterns[r.pattern_rule] = patterns.get(r.pattern_rule, 0) + 1

        print(f"\nPattern Distribution:")
        for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
            print(f"  {pattern}: {count}")

        # Task difficulty
        print(f"\nTask Difficulty (fillable cells):")
        fillables = [r.fillable_cells for r in results]
        print(f"  Mean: {np.mean(fillables):.0f}")
        print(f"  Median: {np.median(fillables):.0f}")
        print(f"  Min: {np.min(fillables)}")
        print(f"  Max: {np.max(fillables)}")

        print(f"\n{'='*80}")


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
    results = run_real_arc_experiments(
        selected_task_file,
        arc_data_dir,
        output_dir,
        max_rollouts=50,
        iterations_per_rollout=20,
        timeout_sec=30.0
    )

    # Save results
    save_results(results, output_dir)
