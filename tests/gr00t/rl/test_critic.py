"""Tests for the BroNet critic ensemble (CPU)."""

from gr00t.rl.critic import BroNetEnsemble, QEnsemble, TargetNetwork, VEnsemble, td_loss
import torch


def test_ensemble_shapes_and_member_independence():
    torch.manual_seed(0)
    net = BroNetEnsemble(input_dim=6, hidden_dim=32, depth=2, num_members=5)
    x = torch.randn(7, 6)
    out = net(x)
    assert out.shape == (5, 7)
    # Different initialisations -> different predictions per member.
    assert out.std(0).mean() > 1e-3
    # Per-member inputs are accepted too.
    assert net(torch.randn(5, 7, 6)).shape == (5, 7)


def test_member_matches_single_mlp():
    """Member m of the batched ensemble equals the same BroNet evaluated on its own."""
    torch.manual_seed(0)
    net = BroNetEnsemble(input_dim=3, hidden_dim=8, depth=1, num_members=3)
    x = torch.randn(4, 3)
    m = 2
    ln = torch.nn.functional.layer_norm

    def lin(layer, h):
        return h @ layer.weight[m] + layer.bias[m]

    def norm(layer, h):
        return ln(h, h.shape[-1:]) * layer.weight[m] + layer.bias[m]

    h = torch.relu(norm(net.proj_norm, lin(net.proj, x)))
    b = net.blocks[0]
    r = torch.relu(norm(b["norm1"], lin(b["linear1"], h)))
    h = h + norm(b["norm2"], lin(b["linear2"], r))
    ref = lin(net.head, h).squeeze(-1)
    assert torch.allclose(net(x)[m], ref, atol=1e-5)


def test_q_and_v_fit_regression_target():
    torch.manual_seed(0)
    q = QEnsemble(prefix_dim=4, state_dim=2, action_dim=6, hidden_dim=64, depth=1, num_members=3)
    opt = torch.optim.Adam(q.parameters(), lr=3e-3)
    prefix, state, action = torch.randn(256, 4), torch.randn(256, 2), torch.randn(256, 3, 2)
    target = -(action.reshape(256, -1) ** 2).sum(-1) + prefix[:, 0]
    for _ in range(400):
        loss = td_loss(q(prefix, state, action), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 * target.var().item()
    v = VEnsemble(prefix_dim=4, state_dim=2, hidden_dim=16, depth=1, num_members=2)
    assert v(prefix, state).shape == (2, 256)


def test_target_network_polyak():
    torch.manual_seed(0)
    online = QEnsemble(2, 1, 2, hidden_dim=8, depth=1, num_members=2)
    tgt = TargetNetwork(online, decay=0.9)
    p_online = next(online.parameters())
    p_target = next(tgt.module.parameters())
    before = p_target.detach().clone()
    with torch.no_grad():
        p_online.add_(1.0)
    tgt.update(online)
    assert torch.allclose(p_target, 0.9 * before + 0.1 * p_online.detach(), atol=1e-6)
    assert not p_target.requires_grad
