import torch

from vbo_mi.models import ActionNet, HelperCritic, SurrogateNet


def test_model_shapes():
    b, d, seed_dim = 12, 10, 16
    lower = torch.full((d,), -1.0)
    upper = torch.full((d,), 1.0)
    seeds = torch.randn(b, seed_dim)
    action = ActionNet(seed_dim, d, lower, upper)
    helper = HelperCritic(d)
    surrogate = SurrogateNet(d)

    x = action(seeds)
    y = surrogate(x)
    scores = helper(x, y)
    assert x.shape == (b, d)
    assert y.shape == (b, 1)
    assert scores.shape == (b, b)


def test_action_bounds():
    d = 6
    lower = torch.tensor([-2., -1., 0., 1., 2., 3.])
    upper = torch.tensor([2., 1., 4., 5., 6., 7.])
    model = ActionNet(8, d, lower, upper)
    x = model(torch.randn(64, 8))
    assert torch.all(x >= lower - 1e-6)
    assert torch.all(x <= upper + 1e-6)
