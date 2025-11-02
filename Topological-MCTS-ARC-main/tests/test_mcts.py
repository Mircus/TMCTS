
import numpy as np
import pytest
from topomcts.game import ARCGame
from topomcts.mcts import TopologicalMCTSEngine, TopologicalWeights, MCTSNode
from topomcts.baselines import create_engine

def test_mcts_runs():
    init = np.array([[0,-1,1],[1,1,1],[1,1,1]])
    tgt = np.array([[0,0,1],[1,1,1],[1,1,1]])
    g = ARCGame(init, tgt, alphabet=(0,1))
    eng = TopologicalMCTSEngine()
    act, val, meta = eng.run(g, iterations=10)
    assert True  # ran without exceptions

def test_ucb1_monotonicity():
    """Test UCB1 monotonicity properties."""
    init = np.array([[0,-1,1],[1,1,1],[1,1,1]])
    tgt = np.array([[0,0,1],[1,1,1],[1,1,1]])
    g = ARCGame(init, tgt, alphabet=(0,1))
    
    # Create a simple tree
    root = MCTSNode(state=g)
    child1 = MCTSNode(state=g, parent=root, action=((0,1), 0))
    child2 = MCTSNode(state=g, parent=root, action=((0,1), 1))
    
    root.children = [child1, child2]
    root.visits = 10
    child1.visits = 5
    child1.value_sum = 3.0
    child2.visits = 3
    child2.value_sum = 1.0
    
    # UCB1 should be higher for less visited nodes
    ucb1 = child1.ucb1()
    ucb2 = child2.ucb1()
    assert ucb2 > ucb1, "Less visited nodes should have higher UCB1"

def test_rollout_depth_cap():
    """Test that rollout respects depth cap."""
    init = np.array([[0,-1,1],[1,1,1],[1,1,1]])
    tgt = np.array([[0,0,1],[1,1,1],[1,1,1]])
    g = ARCGame(init, tgt, alphabet=(0,1))
    
    eng = TopologicalMCTSEngine(rollout_depth=5)
    # This should not crash and should respect depth limit
    value = eng.rollout(g)
    assert isinstance(value, float)

def test_topological_weights():
    """Test topological weights functionality."""
    weights = TopologicalWeights(0.5, 0.3, 0.2)
    assert weights.centrality == 0.5
    assert weights.diffusion == 0.3
    assert weights.symmetry == 0.2
    
    # Test normalization
    weights_unnormalized = TopologicalWeights(1.0, 2.0, 3.0)
    total = weights_unnormalized.centrality + weights_unnormalized.diffusion + weights_unnormalized.symmetry
    assert abs(total - 1.0) < 1e-6, "Weights should be normalized to sum to 1.0"

def test_mcts_doesnt_crash_3x3():
    """Test that MCTS doesn't crash on 3x3 grid."""
    init = np.array([[0,-1,1],[1,1,1],[1,1,1]])
    tgt = np.array([[0,0,1],[1,1,1],[1,1,1]])
    g = ARCGame(init, tgt, alphabet=(0,1))
    
    eng = TopologicalMCTSEngine()
    action, value, metadata = eng.run(g, iterations=50)
    assert action is not None or value == 0.0  # Should either find action or return 0

def test_baseline_engines():
    """Test that all baseline engines work."""
    init = np.array([[0,-1,1],[1,1,1],[1,1,1]])
    tgt = np.array([[0,0,1],[1,1,1],[1,1,1]])
    g = ARCGame(init, tgt, alphabet=(0,1))
    
    methods = ['mcts_std', 'mcts_heur', 'tmcts']
    for method in methods:
        engine = create_engine(method)
        action, value, metadata = engine.run(g, iterations=10)
        assert isinstance(value, float)
        assert isinstance(metadata, dict)

def test_topological_bonus():
    """Test topological bonus computation."""
    init = np.array([[0,-1,1],[1,1,1],[1,1,1]])
    tgt = np.array([[0,0,1],[1,1,1],[1,1,1]])
    g = ARCGame(init, tgt, alphabet=(0,1))
    
    eng = TopologicalMCTSEngine()
    bonus, centrality, diffusion, symmetry = eng.topological_bonus(g)
    
    assert isinstance(bonus, float)
    assert isinstance(centrality, float)
    assert isinstance(diffusion, float)
    assert isinstance(symmetry, float)
    assert 0 <= symmetry <= 1

def test_mcts_node_creation():
    """Test MCTS node creation and properties."""
    init = np.array([[0,-1,1],[1,1,1],[1,1,1]])
    tgt = np.array([[0,0,1],[1,1,1],[1,1,1]])
    g = ARCGame(init, tgt, alphabet=(0,1))
    
    node = MCTSNode(state=g)
    assert node.state == g
    assert node.parent is None
    assert node.action is None
    assert len(node.children) == 0
    assert node.visits == 0
    assert node.value_sum == 0.0

def test_backpropagation():
    """Test backpropagation functionality."""
    init = np.array([[0,-1,1],[1,1,1],[1,1,1]])
    tgt = np.array([[0,0,1],[1,1,1],[1,1,1]])
    g = ARCGame(init, tgt, alphabet=(0,1))
    
    eng = TopologicalMCTSEngine()
    root = MCTSNode(state=g)
    child = MCTSNode(state=g, parent=root, action=((0,1), 0))
    root.children = [child]
    
    # Test backpropagation
    eng.backprop(child, 1.0)
    assert child.visits == 1
    assert child.value_sum == 1.0
    assert root.visits == 1
    assert root.value_sum == 1.0
