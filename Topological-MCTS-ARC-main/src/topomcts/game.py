
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional

class ARCGame:
    """
    Minimal ARC-style grid completion game.
    State: integer grid with -1 for missing cells.
    Actions: fill one missing cell with a symbol from alphabet.
    Terminal: no missing cells remain.
    Reward: 1.0 if matches target exactly, else 0.0 (placeholder).
    """
    def __init__(self, initial: np.ndarray, target: np.ndarray, alphabet=(0,1,2,3,4)):
        assert initial.shape == target.shape, "Initial and target shapes must match"
        self.initial = initial.copy()
        self.target = target.copy()
        self.alphabet = tuple(alphabet)

    def missing_positions(self) -> List[Tuple[int,int]]:
        return list(zip(*np.where(self.initial < 0)))

    def is_terminal(self) -> bool:
        return len(self.missing_positions()) == 0

    def legal_actions(self) -> List[Tuple[Tuple[int,int], int]]:
        acts = []
        for (i, j) in self.missing_positions():
            for a in self.alphabet:
                acts.append(((i, j), a))
        return acts

    def step(self, action: Tuple[Tuple[int,int], int]) -> "ARCGame":
        (i, j), a = action
        nxt = self.initial.copy()
        assert nxt[i, j] < 0, "Position already filled"
        nxt[i, j] = a
        return ARCGame(nxt, self.target, self.alphabet)

    def reward(self) -> float:
        # Placeholder quality metric: exact match → 1, else 0
        return float(np.array_equal(self.initial, self.target))

    def clone(self) -> "ARCGame":
        return ARCGame(self.initial.copy(), self.target.copy(), self.alphabet)

    @staticmethod
    def quality(initial: np.ndarray, target: np.ndarray) -> float:
        # Normalized matching score
        if initial.size == 0:
            return 1.0  # Empty grids are considered perfect matches
        
        # Only consider non-missing cells for quality
        mask = initial >= 0  # Non-missing cells
        if mask.sum() == 0:
            return 1.0  # All cells are missing, consider perfect
        
        ok = (initial[mask] == target[mask]).sum()
        return ok / mask.sum()
