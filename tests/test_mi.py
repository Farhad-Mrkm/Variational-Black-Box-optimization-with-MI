import torch

from vbo_mi.mi import dv_mi_bound, action_loss


def test_dv_bound_is_finite():
    scores = torch.randn(16, 16)
    value = dv_mi_bound(scores)
    assert torch.isfinite(value)


def test_action_loss_propagates_to_scores():
    scores = torch.randn(8, 8, requires_grad=True)
    exploitation = torch.tensor(0.5, requires_grad=True)
    loss = action_loss(scores, exploitation, beta=100.0)
    loss.backward()
    assert scores.grad is not None
    assert exploitation.grad is not None
