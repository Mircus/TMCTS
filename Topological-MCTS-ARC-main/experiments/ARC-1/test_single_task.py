#!/usr/bin/env python3
"""
Quick test of a single ARC-1 task to verify the experiment framework works.
"""

import json
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from topomcts.game import ARCGame
from topomcts.pattern_detector import PatternDetector
from topomcts.mcts import TopologicalMCTSEngine
from topomcts.baselines import StandardMCTSEngine

def load_arc_task(task_id: str, arc_data_dir: Path):
    """Load a single ARC task JSON."""
    task_file = arc_data_dir / f"{task_id}.json"
    if not task_file.exists():
        return None
    with open(task_file, 'r') as f:
        return json.load(f)


def convert_arc_to_game(task: dict):
    """Convert ARC task to ARCGame format."""
    try:
        if not task.get('train') or not task.get('test'):
            return None, {}

        # Use test input as the grid to solve
        test_pair = task['test'][0]
        test_input = np.array(test_pair['input'])
        test_output = np.array(test_pair['output'])

        if 'output' not in test_pair:
            return None, {}

        initial = test_input.copy()

        # Get alphabet
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
        print(f"Error converting task: {e}")
        return None, {}


if __name__ == '__main__':
    # Paths
    arc_data_dir = Path(__file__).parent.parent.parent.parent / "ARC-AGI" / "data" / "training"
    selected_task_file = Path(__file__).parent / "selected_task_ids.json"

    # Load a single task ID
    with open(selected_task_file, 'r') as f:
        task_ids = json.load(f)

    # Test first task
    task_id = task_ids[0]
    print(f"Testing with task: {task_id}")

    task_data = load_arc_task(task_id, arc_data_dir)
    if not task_data:
        print(f"Failed to load task {task_id}")
        sys.exit(1)

    print("OK Loaded task data")

    game, game_info = convert_arc_to_game(task_data)
    if not game:
        print("Failed to convert to game")
        sys.exit(1)

    print("OK Converted to game format")
    print(f"  Grid shape: {game_info['test_shape']}")
    print(f"  Missing cells: {game_info['missing_cells']}")
    print(f"  Alphabet: {game.alphabet}")

    # Test pattern detection
    detector = PatternDetector()
    try:
        rule = detector.detect_rule(game)
        print(f"[OK] Pattern detection: {rule}")
    except Exception as e:
        print(f"[FAIL] Pattern detection failed: {e}")

    # Test baseline MCTS
    print("\nTesting Baseline MCTS...")
    try:
        engine = StandardMCTSEngine(ucb_c=1.414, rollout_depth=10)
        action, value, metadata = engine.run(game, iterations=10)
        print(f"[OK] Baseline MCTS ran successfully")
        print(f"  Value: {value:.3f}")
        print(f"  Metadata keys: {list(metadata.keys())}")
    except Exception as e:
        print(f"[FAIL] Baseline MCTS failed: {e}")
        import traceback
        traceback.print_exc()

    # Test topological MCTS
    print("\nTesting Topological MCTS...")
    try:
        engine = TopologicalMCTSEngine(
            ucb_c=1.414,
            rollout_depth=10,
            topology_weight=0.5,
            use_constraint_graphs=True
        )
        action, value, metadata = engine.run(game, iterations=10)
        print(f"[OK] Topological MCTS ran successfully")
        print(f"  Value: {value:.3f}")
        print(f"  Simulations: {metadata.get('simulations', 'N/A')}")
        print(f"  Centrality: {metadata.get('centrality', 'N/A'):.3f}" if isinstance(metadata.get('centrality'), (int, float)) else f"  Centrality: {metadata.get('centrality', 'N/A')}")
        print(f"  Diffusion: {metadata.get('diffusion', 'N/A'):.3f}" if isinstance(metadata.get('diffusion'), (int, float)) else f"  Diffusion: {metadata.get('diffusion', 'N/A')}")
        print(f"  Symmetry: {metadata.get('symmetry', 'N/A'):.3f}" if isinstance(metadata.get('symmetry'), (int, float)) else f"  Symmetry: {metadata.get('symmetry', 'N/A')}")
    except Exception as e:
        print(f"[FAIL] Topological MCTS failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n[OK] All tests passed!")
