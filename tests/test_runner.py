import torch

from vbo_mi import VBOConfig, VBOMI
from vbo_mi.objectives import BlackBoxObjective


def test_query_budget_and_black_box_interface():
    calls = {"n": 0}

    def objective_fn(x):
        # Deliberately reject autograd-enabled inputs to catch accidental oracle gradients.
        assert not x.requires_grad
        calls["n"] += x.shape[0]
        return -(x ** 2).sum(dim=1, keepdim=True)

    objective = BlackBoxObjective(
        objective_fn,
        lower=torch.full((3,), -1.0),
        upper=torch.full((3,), 1.0),
        name="quadratic",
    )
    cfg = VBOConfig(seed=0, seed_dim=4, action_hidden_dim=16, helper_hidden_dim=8, surrogate_hidden_dim=16,
                    ka=1, kb=1, ks=1, surrogate_pretrain_steps=2)
    result = VBOMI(objective, cfg).run(batch_size=8, warmup_iterations=2, main_iterations=3)
    assert calls["n"] == (2 + 3) * 8
    assert result.evaluation_counts[-1] == 40
    assert result.best_values.shape == (3,)
