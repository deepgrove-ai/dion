import math
from fractions import Fraction
from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh
from torch.optim.optimizer import ParamsT

from .muon import Muon


MuonPExponent = Union[str, int, float, Fraction, Tuple[int, int]]


_DEFAULT_COEFFICIENTS = {
    Fraction(1, 2): (3.0, -0.795918),
    Fraction(1, 3): (0.66, None),
    Fraction(1, 5): (0.2, None),
    Fraction(1, 7): (1.0 / 7.0, None),
    Fraction(3, 5): (0.2, None),
    Fraction(1, 15): (1.0 / 15.0, None),
    Fraction(13, 15): (1.0 / 15.0, None),
}


def _parse_exponent(exponent: MuonPExponent) -> Fraction:
    if isinstance(exponent, Fraction):
        value = exponent
    elif isinstance(exponent, tuple):
        if len(exponent) != 2:
            raise ValueError(
                f"Exponent tuple must be (numerator, denominator), got {exponent!r}"
            )
        value = Fraction(exponent[0], exponent[1])
    elif isinstance(exponent, str):
        value = Fraction(exponent.strip())
    elif isinstance(exponent, float):
        value = Fraction(str(exponent)).limit_denominator(1000)
    else:
        value = Fraction(exponent)

    if not (0 < value < 1):
        raise ValueError(f"MuonP exponent must satisfy 0 < p < 1, got {value}")
    return value


def resolve_muon_p_coefficients(
    exponent: MuonPExponent,
    c: Optional[float] = None,
    d: Optional[float] = None,
) -> Tuple[Fraction, float, Optional[float]]:
    exponent = _parse_exponent(exponent)
    default_c, default_d = _DEFAULT_COEFFICIENTS.get(
        exponent, (1.0 / exponent.denominator, None)
    )
    c = default_c if c is None else c
    d = default_d if d is None else d

    if c <= 0:
        raise ValueError(f"MuonP coefficient c must be positive, got {c}")
    if exponent == Fraction(1, 2) and d is None:
        raise ValueError("MuonP exponent 1/2 requires coefficient d")
    if exponent != Fraction(1, 2) and (
        exponent.numerator % 2 == 0 or exponent.denominator % 2 == 0
    ):
        raise NotImplementedError(
            "MuonP currently supports exponent 1/2 and rational exponents "
            "with odd numerator and odd denominator."
        )
    return exponent, float(c), None if d is None else float(d)


def _spectral_odd_power(X: Tensor, power: int) -> Tensor:
    if power < 1 or power % 2 == 0:
        raise ValueError(f"Expected a positive odd spectral power, got {power}")
    if power == 1:
        return X

    gram = X @ X.mT
    out = X
    for _ in range((power - 1) // 2):
        out = gram @ out
    return out


def _muon_p_iteration(
    X: Tensor,
    Y: Tensor,
    exponent: Fraction,
    c: float,
    d: Optional[float],
) -> Tensor:
    if exponent == Fraction(1, 2):
        # Upstream Muon-P poly_half:
        # y - ((y y.T)^2 - x x.T) @ (c x + d y)
        YY = Y @ Y.mT
        return Y - (YY @ YY - X @ X.mT) @ (c * X + d * Y)

    numerator = exponent.numerator
    denominator = exponent.denominator
    if numerator % 2 == 0 or denominator % 2 == 0:
        raise NotImplementedError(
            "MuonP currently supports exponent 1/2 and rational exponents "
            "with odd numerator and odd denominator."
        )

    return Y - c * (
        _spectral_odd_power(Y, denominator) - _spectral_odd_power(X, numerator)
    )


def muon_p_spectral_power(
    G: Tensor,
    epsilon: Union[float, Tensor] = 1e-7,
    exponent: MuonPExponent = "1/3",
    steps: int = 6,
    c: Optional[float] = None,
    d: Optional[float] = None,
    normalize_output: bool = False,
) -> Tensor:
    """
    Approximate the spectral fractional power update ``U S**p V.T``.

    ``G`` is first scaled to Frobenius norm at most one, matching the released
    Muon-P implementation. If ``normalize_output`` is true, the result is scaled
    to unit elementwise RMS, matching the optimizer-side scaling used upstream.
    """
    if steps < 1:
        raise ValueError(f"MuonP steps must be >= 1, got {steps}")

    exponent, c, d = resolve_muon_p_coefficients(exponent, c=c, d=d)

    X = G.to(dtype=torch.bfloat16)
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)
    Y = X

    for _ in range(steps):
        Y = _muon_p_iteration(X, Y, exponent, c, d)

    if normalize_output:
        rms_scale = math.sqrt(Y.size(-2) * Y.size(-1))
        Y = Y * (rms_scale / (Y.norm(dim=(-2, -1), keepdim=True) + epsilon))

    if transposed:
        Y = Y.mT
    return Y.to(dtype=G.dtype)


def make_muon_p_spectral_power(
    exponent: MuonPExponent = "1/3",
    steps: int = 6,
    c: Optional[float] = None,
    d: Optional[float] = None,
    normalize_output: bool = False,
) -> Callable[[Tensor, Union[float, Tensor]], Tensor]:
    exponent = _parse_exponent(exponent)

    def _fn(G: Tensor, epsilon: Union[float, Tensor] = 1e-7) -> Tensor:
        return muon_p_spectral_power(
            G,
            epsilon=epsilon,
            exponent=exponent,
            steps=steps,
            c=c,
            d=d,
            normalize_output=normalize_output,
        )

    return _fn


class MuonP(Muon):
    """
    Distributed MuonP optimizer: Muon with fractional spectral powers.

    The default exponent and coefficients follow the released Princeton PLI
    Muon-P code: exponent ``1/3``, six polynomial iterations, and ``c=0.66``.
    By default the transformed update is normalized to unit elementwise RMS,
    which matches the update scaling in the authors' optimizer.
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        mu: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        cautious_wd: bool = False,
        epsilon: float = 1e-8,
        nesterov: bool = True,
        adjust_lr: Optional[str] = None,
        flatten: bool = False,
        exponent: MuonPExponent = "1/3",
        steps: int = 6,
        c: Optional[float] = None,
        d: Optional[float] = None,
        normalize_update: bool = True,
        mixed_precision_config: Optional[Any] = None,
    ):
        newton_schulz_func = make_muon_p_spectral_power(
            exponent=exponent,
            steps=steps,
            c=c,
            d=d,
            normalize_output=normalize_update,
        )
        super().__init__(
            params=params,
            distributed_mesh=distributed_mesh,
            lr=lr,
            mu=mu,
            betas=betas,
            weight_decay=weight_decay,
            cautious_wd=cautious_wd,
            epsilon=epsilon,
            nesterov=nesterov,
            adjust_lr=adjust_lr,
            flatten=flatten,
            use_gram_newton_schulz=False,
            use_triton=False,
            use_polar_express=False,
            newton_schulz_func=newton_schulz_func,
            mixed_precision_config=mixed_precision_config,
        )
