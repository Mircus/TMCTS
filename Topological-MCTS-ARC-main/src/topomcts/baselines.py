from __future__ import annotations
import math, random
import numpy as np
from typing import Any, Tuple, Dict
from .game import ARCGame
from .mcts import MCTSNode, TopologicalMCTSEngine, TopologicalWeights
from .complex import SimplicialComplex

class StandardMCTSEngine:
    """Vanilla MCTS with UCB1 only (no topological bonus)."""
    
    def __init__(self, ucb_c: float = 1.414, rollout_depth: int = 10):
        self.ucb_c = float(ucb_c)
        self.rollout_depth = int(rollout_depth)

    def select(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCB1 only."""
        cur = node
        while cur.children:
            cur = max(cur.children, key=lambda c: c.ucb1(c=self.ucb_c))
        return cur

    def expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by adding all possible children."""
        if node.state.is_terminal():
            return node
        
        actions = node.state.legal_actions()
        random.shuffle(actions)
        
        for action in actions:
            child = MCTSNode(state=node.state.step(action), parent=node, action=action)
            node.children.append(child)
        
        return random.choice(node.children) if node.children else node

    def rollout(self, state: ARCGame) -> float:
        """Random rollout to estimate state value."""
        cur = state.clone()
        depth = 0
        
        while (not cur.is_terminal()) and depth < self.rollout_depth:
            actions = cur.legal_actions()
            if not actions:
                break
            cur = cur.step(random.choice(actions))
            depth += 1
        
        return cur.reward()

    def backprop(self, node: MCTSNode, value: float) -> None:
        """Backpropagate value up the tree."""
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value_sum += value
            cur = cur.parent

    def run(self, root_state: ARCGame, iterations: int = 200) -> Tuple[Any, float, Dict]:
        """Run MCTS search and return best action, value, and metadata."""
        root = MCTSNode(state=root_state)
        
        for _ in range(iterations):
            leaf = self.select(root)
            child = self.expand(leaf)
            value = self.rollout(child.state)
            self.backprop(child, value)
        
        if not root.children:
            return None, 0.0, {}
        
        best = max(root.children, key=lambda c: c.visits)
        best_value = best.value_sum / max(1, best.visits)
        
        metadata = {
            'centrality': 0.0,
            'diffusion': 0.0,
            'symmetry': 0.0,
            'simulations': iterations,
            'total_visits': root.visits
        }
        
        return best.action, best_value, metadata

class HeuristicMCTSEngine(StandardMCTSEngine):
    """MCTS with simple domain heuristic bonus."""
    
    def __init__(self, ucb_c: float = 1.414, rollout_depth: int = 10, heuristic_weight: float = 0.3):
        super().__init__(ucb_c, rollout_depth)
        self.heuristic_weight = float(heuristic_weight)

    def heuristic_bonus(self, state: ARCGame) -> float:
        """Simple heuristic: prefer states with fewer missing cells."""
        missing_count = len(state.missing_positions())
        total_cells = state.initial.size
        return 1.0 - (missing_count / total_cells)

    def select(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCB1 + heuristic bonus."""
        cur = node
        while cur.children:
            def score(child: MCTSNode) -> float:
                base = child.ucb1(c=self.ucb_c)
                heuristic = self.heuristic_weight * self.heuristic_bonus(child.state)
                return base + heuristic
            cur = max(cur.children, key=score)
        return cur

class NeuralMCTSEngine(StandardMCTSEngine):
    """MCTS with neural network policy/value (stub implementation)."""
    
    def __init__(self, ucb_c: float = 1.414, rollout_depth: int = 10, neural_weight: float = 0.5):
        super().__init__(ucb_c, rollout_depth)
        self.neural_weight = float(neural_weight)

    def neural_policy(self, state: ARCGame) -> float:
        """Stub neural policy - returns random value for now."""
        # In real implementation, this would use a trained neural network
        return random.random()

    def select(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCB1 + neural policy."""
        cur = node
        while cur.children:
            def score(child: MCTSNode) -> float:
                base = child.ucb1(c=self.ucb_c)
                neural = self.neural_weight * self.neural_policy(child.state)
                return base + neural
            cur = max(cur.children, key=score)
        return cur

class PersistentHomologyEngine(StandardMCTSEngine):
    """MCTS with persistent homology features (stub implementation)."""
    
    def __init__(self, ucb_c: float = 1.414, rollout_depth: int = 10, ph_weight: float = 0.4):
        super().__init__(ucb_c, rollout_depth)
        self.ph_weight = float(ph_weight)

    def ph_features(self, state: ARCGame) -> float:
        """Stub persistent homology features."""
        # In real implementation, this would compute PH features using Gudhi/Giotto-TDA
        comp = SimplicialComplex.from_grid(state.initial)
        return comp.centrality_score()  # Placeholder

    def select(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCB1 + PH features."""
        cur = node
        while cur.children:
            def score(child: MCTSNode) -> float:
                base = child.ucb1(c=self.ucb_c)
                ph = self.ph_weight * self.ph_features(child.state)
                return base + ph
            cur = max(cur.children, key=score)
        return cur

class GNNTransferEngine(TopologicalMCTSEngine):
    """MCTS with GNN-based transfer learning (stub implementation)."""
    
    def __init__(self, weights: TopologicalWeights = None, ucb_c: float = 1.414, 
                 rollout_depth: int = 10, topology_weight: float = 0.5, transfer_weight: float = 0.3):
        super().__init__(weights, ucb_c, rollout_depth, topology_weight)
        self.transfer_weight = float(transfer_weight)

    def transfer_bonus(self, state: ARCGame) -> float:
        """Stub transfer learning bonus."""
        # In real implementation, this would use learned GNN embeddings
        return random.random()

    def select(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCB1 + topological + transfer bonus."""
        cur = node
        while cur.children:
            def score(child: MCTSNode) -> float:
                base = child.ucb1(c=self.ucb_c)
                topo_bonus, _, _, _ = self.topological_bonus(child.state)
                transfer = self.transfer_weight * self.transfer_bonus(child.state)
                return base + self.topology_weight * topo_bonus + transfer
            cur = max(cur.children, key=score)
        return cur

# Factory function for creating engines
def create_engine(method: str, **kwargs) -> Any:
    """Create MCTS engine based on method name."""
    engines = {
        'mcts_std': StandardMCTSEngine,
        'mcts_heur': HeuristicMCTSEngine,
        'mcts_neural': NeuralMCTSEngine,
        'ph_features': PersistentHomologyEngine,
        'tmcts': TopologicalMCTSEngine,
        'gnn_transfer': GNNTransferEngine
    }
    
    if method not in engines:
        raise ValueError(f"Unknown method: {method}. Available: {list(engines.keys())}")
    
    return engines[method](**kwargs)
