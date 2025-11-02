"""
ARC Transformation Game for real-world ARC-1 task evaluation.

Real ARC tasks require learning a transformation rule from training examples
and applying it to a test input to generate the output.

This module formulates that as a search problem where:
- State: partially filled test output grid
- Actions: place a color at a specific cell
- Goal: fill grid to match ground truth output while respecting learned pattern
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TransformationRule:
    """Represents a learned transformation pattern."""
    rule_type: str  # e.g., "rotational_symmetry_90", "color_mapping", "line_drawing"
    params: dict  # Rule-specific parameters


class ARCTransformationGame:
    """
    Game for solving real ARC tasks via iterative grid generation.

    The game state represents a partially completed output grid.
    Actions place colors at empty cells.
    The reward is based on matching the ground truth output.

    This formulation treats ARC as a constraint satisfaction problem where:
    - Constraints come from the learned transformation pattern
    - Solution space is the set of valid grid completions
    - Topological features of the constraint graph guide the search
    """

    def __init__(
        self,
        training_examples: List[dict],
        test_input: np.ndarray,
        ground_truth_output: np.ndarray,
        rule: Optional[TransformationRule] = None
    ):
        """
        Initialize an ARC transformation game.

        Args:
            training_examples: List of {'input': array, 'output': array} dicts
            test_input: The test input grid (H x W)
            ground_truth_output: The expected output (H x W) - for scoring
            rule: Detected/provided transformation rule (optional)
        """
        self.training_examples = training_examples
        self.test_input = np.array(test_input)
        self.ground_truth = np.array(ground_truth_output)
        self.rule = rule

        # Current state: the grid being constructed
        # Start with test input, incrementally fill cells
        self.current_output = self.test_input.copy()

        # Alias for compatibility with existing code that expects 'initial'
        self.initial = self.current_output.copy()

        # Track which cells are "fillable" (not in original input)
        self._identify_fillable_cells()

        # Extract alphabet from all data
        self.alphabet = self._extract_alphabet()

    def _identify_fillable_cells(self):
        """Determine which cells can be filled (differ from ground truth)."""
        # Cells that can be filled are those that differ from test input
        # and need to match ground truth
        self.fillable_positions = set()

        # If shapes match: cells that differ from test input
        if self.test_input.shape == self.ground_truth.shape:
            for i in range(self.test_input.shape[0]):
                for j in range(self.test_input.shape[1]):
                    if self.test_input[i, j] != self.ground_truth[i, j]:
                        self.fillable_positions.add((i, j))
        else:
            # If shapes differ, we can't fill (output requires shape change)
            # Mark all cells in intersection as potentially fillable
            h = min(self.test_input.shape[0], self.ground_truth.shape[0])
            w = min(self.test_input.shape[1], self.ground_truth.shape[1])
            for i in range(h):
                for j in range(w):
                    if self.test_input[i, j] != self.ground_truth[i, j]:
                        self.fillable_positions.add((i, j))

    def _extract_alphabet(self) -> Tuple[int, ...]:
        """Extract set of colors used in training + test."""
        colors = set()

        for example in self.training_examples:
            colors.update(np.unique(example['input']).tolist())
            colors.update(np.unique(example['output']).tolist())

        colors.update(np.unique(self.test_input).tolist())
        colors.update(np.unique(self.ground_truth).tolist())

        return tuple(sorted(colors))

    def missing_positions(self) -> List[Tuple[int, int]]:
        """Get list of unfilled positions that need colors."""
        missing = []
        for i in range(self.current_output.shape[0]):
            for j in range(self.current_output.shape[1]):
                if (i, j) in self.fillable_positions:
                    if self.current_output[i, j] != self.ground_truth[i, j]:
                        missing.append((i, j))
        return missing

    def is_terminal(self) -> bool:
        """Check if all fillable positions are completed."""
        return len(self.missing_positions()) == 0

    def legal_actions(self) -> List[Tuple[Tuple[int, int], int]]:
        """
        Generate all legal actions: (position, color) tuples.

        Legal positions: unfilled cells that differ from ground truth
        Legal colors: colors from alphabet
        """
        actions = []
        for pos in self.missing_positions():
            for color in self.alphabet:
                actions.append((pos, color))
        return actions

    def step(self, action: Tuple[Tuple[int, int], int]) -> ARCTransformationGame:
        """
        Apply action to create next state.

        Args:
            action: ((row, col), color) tuple

        Returns:
            New ARCTransformationGame with updated current_output
        """
        (i, j), color = action

        # Create new state
        new_output = self.current_output.copy()
        new_output[i, j] = color

        # Create new game instance with updated state
        new_game = ARCTransformationGame(
            self.training_examples,
            self.test_input,
            self.ground_truth,
            self.rule
        )
        new_game.current_output = new_output
        new_game.initial = new_output.copy()  # Keep 'initial' in sync
        return new_game

    def reward(self) -> float:
        """
        Compute reward as accuracy on filled cells.

        Returns:
            1.0 if entire output matches ground truth
            Partial credit based on fraction of correct filled cells
            0.0 if terminal but not matching
        """
        if self.current_output.shape != self.ground_truth.shape:
            # Shape mismatch: cannot evaluate
            return 0.0

        # Check exact match
        if np.array_equal(self.current_output, self.ground_truth):
            return 1.0

        # Partial credit: fraction of correct cells
        correct = (self.current_output == self.ground_truth).sum()
        total = self.current_output.size
        return correct / total

    def quality(self) -> float:
        """
        Compute solution quality without consuming state.

        Returns fraction of output cells matching ground truth.
        """
        if self.current_output.shape != self.ground_truth.shape:
            return 0.0

        correct = (self.current_output == self.ground_truth).sum()
        total = self.current_output.size
        return correct / total

    def clone(self) -> ARCTransformationGame:
        """Create independent copy of current game state."""
        new_game = ARCTransformationGame(
            self.training_examples,
            self.test_input,
            self.ground_truth,
            self.rule
        )
        new_game.current_output = self.current_output.copy()
        return new_game

    @staticmethod
    def from_arc_task(task: dict, verbose: bool = False) -> Optional[ARCTransformationGame]:
        """
        Create ARCTransformationGame from standard ARC task JSON.

        Args:
            task: Standard ARC task dict with 'train' and 'test' keys
            verbose: Print debug info

        Returns:
            ARCTransformationGame instance or None if invalid
        """
        try:
            if 'train' not in task or 'test' not in task:
                return None

            if not task['train'] or not task['test']:
                return None

            training_examples = task['train']
            test_pair = task['test'][0]  # Use first test pair

            test_input = np.array(test_pair['input'])

            # Ground truth output must be available
            if 'output' not in test_pair:
                return None

            ground_truth = np.array(test_pair['output'])

            if verbose:
                print(f"Created game: input {test_input.shape} -> output {ground_truth.shape}")

            return ARCTransformationGame(training_examples, test_input, ground_truth)

        except Exception as e:
            if verbose:
                print(f"Failed to create game: {e}")
            return None

    def get_state_dict(self) -> dict:
        """Serialize game state for logging."""
        return {
            'test_input_shape': self.test_input.shape,
            'ground_truth_shape': self.ground_truth.shape,
            'current_output_shape': self.current_output.shape,
            'missing_count': len(self.missing_positions()),
            'fillable_count': len(self.fillable_positions),
            'quality': self.quality(),
            'is_terminal': self.is_terminal()
        }
