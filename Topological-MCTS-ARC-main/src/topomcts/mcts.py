
from __future__ import annotations
import math, random
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, List
from .game import ARCGame
from .complex import SimplicialComplex
from .pattern_detector import PatternDetector
from .constraint_graph import ConstraintGraph

@dataclass
class MCTSNode:
    state: ARCGame
    parent: Optional["MCTSNode"] = None
    action: Optional[Any] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0

    def ucb1(self, c: float = 1.414) -> float:
        """Compute UCB1 score for node selection."""
        if self.visits == 0:
            return float("inf")
        exploit = self.value_sum / self.visits
        explore = c * math.sqrt(math.log(self.parent.visits + 1) / self.visits) if self.parent else 0.0
        return exploit + explore

@dataclass
class TopologicalWeights:
    """Weights for composable topological bonus."""
    centrality: float = 0.4
    diffusion: float = 0.3
    symmetry: float = 0.3
    
    def __post_init__(self):
        # Normalize weights to sum to 1.0
        total = self.centrality + self.diffusion + self.symmetry
        if total > 0:
            self.centrality /= total
            self.diffusion /= total
            self.symmetry /= total

class TopologicalMCTSEngine:
    def __init__(
        self,
        weights: TopologicalWeights = None,
        ucb_c: float = 1.414,
        rollout_depth: int = 10,
        topology_weight: float = 0.5,
        use_constraint_graphs: bool = True
    ):
        self.weights = weights or TopologicalWeights()
        self.ucb_c = float(ucb_c)
        self.rollout_depth = int(rollout_depth)
        self.topology_weight = float(topology_weight)
        self.use_constraint_graphs = use_constraint_graphs
        self.pattern_detector = PatternDetector()
        self._constraint_graph_cache = {}  # Cache graphs to avoid recomputation

    def topological_bonus(self, state: ARCGame) -> Tuple[float, float, float, float]:
        """
        Compute topological bonus from solution space topology.

        Uses constraint graph features:
        - centrality: average cell importance (how constrained cells are)
        - diffusion: algebraic connectivity (solution space fragmentation)
        - symmetry: color importance variance (pattern structure)
        """
        if self.use_constraint_graphs:
            return self._constraint_graph_bonus(state)
        else:
            # Fallback to grid topology for comparison
            return self._grid_topology_bonus(state)

    def _constraint_graph_bonus(self, state: ARCGame) -> Tuple[float, float, float, float]:
        """Compute bonus from solution space topology (constraint graph)."""
        # Detect pattern rule
        rule = self.pattern_detector.detect_rule(state)

        # Build or retrieve cached constraint graph
        # Use grid data hash + rule as cache key
        state_key = (state.initial.tobytes(), rule)
        if state_key not in self._constraint_graph_cache:
            cg = ConstraintGraph(state, rule)
            self._constraint_graph_cache[state_key] = cg
        else:
            cg = self._constraint_graph_cache[state_key]

        # Extract topological features
        features = cg.topological_features()

        # Map features to bonus components:
        # - centrality: average cell importance (higher = more constrained = harder)
        cell_imp = features['cell_importance']
        centrality = sum(cell_imp.values()) / max(1, len(cell_imp)) if cell_imp else 0.0

        # - diffusion: algebraic connectivity (higher = better connected = less fragmented)
        diffusion = features['algebraic_connectivity']

        # - symmetry: variance in color importance (higher = more structured)
        color_imp = features['color_importance']
        if color_imp:
            mean_imp = sum(color_imp.values()) / len(color_imp)
            variance = sum((v - mean_imp) ** 2 for v in color_imp.values()) / len(color_imp)
            symmetry = math.sqrt(variance)  # Std dev of color importance
        else:
            symmetry = 0.0

        # Compose total bonus
        total_bonus = (
            self.weights.centrality * centrality +
            self.weights.diffusion * diffusion +
            self.weights.symmetry * symmetry
        )

        return total_bonus, centrality, diffusion, symmetry

    def _grid_topology_bonus(self, state: ARCGame) -> Tuple[float, float, float, float]:
        """Fallback: compute bonus from grid topology (SimplicialComplex)."""
        comp = SimplicialComplex.from_grid(state.initial)
        centrality, diffusion, symmetry = comp.topological_features()

        total_bonus = (
            self.weights.centrality * centrality +
            self.weights.diffusion * diffusion +
            self.weights.symmetry * symmetry
        )

        return total_bonus, centrality, diffusion, symmetry

    def select(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCB1 + topological bonus."""
        cur = node
        while cur.children:
            def score(child: MCTSNode) -> float:
                base = child.ucb1(c=self.ucb_c)
                topo_bonus, _, _, _ = self.topological_bonus(child.state)
                return base + self.topology_weight * topo_bonus
            cur = max(cur.children, key=score)
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

    def run(self, root_state: ARCGame, iterations: int = 200) -> Tuple[Any, float, dict]:
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
        
        # Compute final topological features for metadata
        _, centrality, diffusion, symmetry = self.topological_bonus(root_state)
        
        metadata = {
            'centrality': centrality,
            'diffusion': diffusion,
            'symmetry': symmetry,
            'simulations': iterations,
            'total_visits': root.visits
        }
        
        return best.action, best_value, metadata
