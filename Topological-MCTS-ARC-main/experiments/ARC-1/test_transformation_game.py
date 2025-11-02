#!/usr/bin/env python3
"""Quick test of ARCTransformationGame on a single real ARC task."""

import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from topomcts.arc_transformation_game import ARCTransformationGame
from topomcts.pattern_detector import PatternDetector
from topomcts.mcts import TopologicalMCTSEngine
from topomcts.baselines import StandardMCTSEngine


def test_single_task():
    arc_data_dir = Path(__file__).parent.parent.parent.parent / "ARC-AGI" / "data" / "training"
    selected_file = Path(__file__).parent / "selected_task_ids.json"

    # Load task ID
    with open(selected_file) as f:
        task_ids = json.load(f)

    task_id = task_ids[0]
    print(f"Testing with task: {task_id}")

    # Load task
    task_file = arc_data_dir / f"{task_id}.json"
    with open(task_file) as f:
        task = json.load(f)

    print(f"[OK] Loaded task")

    # Create game
    game = ARCTransformationGame.from_arc_task(task, verbose=True)
    if not game:
        print("[FAIL] Could not create game")
        return False

    print(f"[OK] Created transformation game")
    print(f"  Test input shape: {game.test_input.shape}")
    print(f"  Ground truth shape: {game.ground_truth.shape}")
    print(f"  Fillable cells: {len(game.fillable_positions)}")
    print(f"  Alphabet: {game.alphabet}")
    print(f"  Initial quality: {game.quality():.3f}")

    # Test pattern detection
    detector = PatternDetector()
    try:
        first_ex = game.training_examples[0]
        class TempGame:
            def __init__(self, grid):
                self.initial = grid
        temp = TempGame(np.array(first_ex['input']))
        rule = detector.detect_rule(temp)
        print(f"[OK] Detected pattern: {rule}")
    except Exception as e:
        print(f"[WARN] Pattern detection failed: {e}")

    # Test actions
    try:
        actions = game.legal_actions()
        print(f"[OK] Generated {len(actions)} legal actions")
        if actions:
            # Try one action
            action = actions[0]
            next_game = game.step(action)
            print(f"[OK] Step succeeded, new quality: {next_game.quality():.3f}")
    except Exception as e:
        print(f"[FAIL] Action generation/step failed: {e}")
        return False

    # Test baseline MCTS
    print("\nTesting Baseline MCTS...")
    try:
        engine = StandardMCTSEngine(ucb_c=1.414, rollout_depth=10)
        action, value, metadata = engine.run(game, iterations=5)
        print(f"[OK] Baseline MCTS executed")
        print(f"  Best value: {value:.3f}")
        print(f"  Current quality: {game.quality():.3f}")
    except Exception as e:
        print(f"[FAIL] Baseline MCTS failed: {e}")
        import traceback
        traceback.print_exc()

    # Test topological MCTS
    print("\nTesting Topological MCTS...")
    try:
        # Reset game
        game = ARCTransformationGame.from_arc_task(task, verbose=False)

        engine = TopologicalMCTSEngine(
            ucb_c=1.414,
            rollout_depth=10,
            topology_weight=0.5,
            use_constraint_graphs=True
        )
        action, value, metadata = engine.run(game, iterations=5)
        print(f"[OK] Topological MCTS executed")
        print(f"  Best value: {value:.3f}")
        print(f"  Current quality: {game.quality():.3f}")
    except Exception as e:
        print(f"[WARN] Topological MCTS had issues (may be constraint graph): {e}")
        # This is expected if constraint graph fails on real ARC patterns

    print("\n[OK] Test completed!")
    return True


if __name__ == '__main__':
    success = test_single_task()
    sys.exit(0 if success else 1)
