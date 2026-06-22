from fractions import Fraction
import math

import torch


def test_muon_p_default_coefficients_match_release():
    from dion import resolve_muon_p_coefficients

    exponent, c, d = resolve_muon_p_coefficients("1/3")
    assert exponent == Fraction(1, 3)
    assert c == 0.66
    assert d is None

    exponent, c, d = resolve_muon_p_coefficients("1/2")
    assert exponent == Fraction(1, 2)
    assert c == 3.0
    assert d == -0.795918


def test_muon_p_one_step_third_matches_polynomial():
    from dion import muon_p_spectral_power

    torch.manual_seed(0)
    g = torch.randn(4, 7)
    eps = 1e-7

    x = g.to(torch.bfloat16)
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    expected = x - 0.66 * (x @ x.mT @ x - x)

    actual = muon_p_spectral_power(
        g,
        epsilon=eps,
        exponent="1/3",
        steps=1,
        c=0.66,
        normalize_output=False,
    )
    torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=0, atol=0)


def test_muon_p_normalize_output_unit_rms():
    from dion import muon_p_spectral_power

    torch.manual_seed(1)
    g = torch.randn(5, 9)
    out = muon_p_spectral_power(g, exponent="1/3", normalize_output=True)

    rms = out.float().norm() / math.sqrt(out.numel())
    torch.testing.assert_close(rms, torch.tensor(1.0), rtol=6e-3, atol=6e-3)


def test_muon_p_supported_exponents_are_finite():
    from dion import muon_p_spectral_power

    torch.manual_seed(2)
    g = torch.randn(3, 6)
    for exponent in ("1/2", "1/3", "1/5", "1/7", "3/5", "1/15", "13/15"):
        out = muon_p_spectral_power(g, exponent=exponent, steps=2)
        assert out.shape == g.shape
        assert torch.isfinite(out).all()


def test_muon_p_cpu_step_changes_param_and_creates_state():
    from dion import MuonP

    torch.manual_seed(3)
    param = torch.nn.Parameter(torch.randn(4, 8))
    opt = MuonP([param], lr=0.01, weight_decay=0.0)

    before = param.detach().clone()
    param.grad = torch.randn_like(param)
    opt.step()

    state = opt.state[param]
    assert "momentum" in state
    assert not torch.equal(param.detach(), before)
