
from __future__ import annotations
import numpy as np
from .game import ARCGame

def success(game: ARCGame) -> float:
    return game.reward()

def quality(game: ARCGame) -> float:
    return ARCGame.quality(game.initial, game.target)
