import importlib.util

import pytest


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch not installed")
def test_actor_value_shapes_and_action_mask() -> None:
    import torch

    from arc_agent.models.ppo_actor_critic import PPOActorCritic

    model = PPOActorCritic(graph_embedding_dim=8, belief_feature_dim=4, hidden_dim=16, num_actions=3)
    graph = torch.zeros((2, 8))
    belief = torch.zeros((2, 4))
    mask = torch.tensor([[1, 0, 1], [0, 1, 1]])
    output = model(graph, belief, mask)
    assert output.logits.shape == (2, 3)
    assert output.value.shape == (2,)
    assert output.logits[0, 1].item() < -1e8
