from math import isfinite
from typing import Callable, Union

import numpy as np

from . import _core
from .types import DType

real = Union[float, int]


def _check_all_uint(size: tuple[int, ...]) -> bool:
    return all(isinstance(i, int) and not isinstance(i, bool) and i > 0 for i in size)


def _ensure_real(value: real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


def _ensure_integer(value: real, name: str) -> int:
    value = _ensure_real(value, name)
    if not value.is_integer():
        raise ValueError(f"{name} must be integer-valued, got {value}")
    return int(value)


def _generate_random_values(
    size: int | tuple[int, ...],
    single_val_generator: Callable[..., Union[float, int, bool]],
    array_generator: Callable[..., np.ndarray],
    func_args: tuple,
) -> Union[float, int, bool, np.ndarray]:
    """Helper function to generate single or multiple random values based on size."""
    if isinstance(size, int) and not isinstance(size, bool):
        if size == 1:
            return single_val_generator(*func_args)
        elif size > 0:  # Handles integers greater than 1
            return array_generator(size, *func_args)
        else:
            raise ValueError(f"Invalid size {size}, expected positive integer")
    elif isinstance(size, tuple):
        if _check_all_uint(size) and len(size) > 0:
            length = int(np.prod(size))
            arr = array_generator(length, *func_args)
            return arr.reshape(size)
        else:
            raise ValueError(
                f"Invalid size {size}, expected tuple of positive integers"
            )
    else:
        # Match original error message style for type
        raise TypeError(
            f"Invalid size type {type(size)}, expected positive integer or tuple of positive integers"
        )


def randexp(
    size: int | tuple[int, ...] = 1, scale: real = 1.0
) -> Union[float, np.ndarray]:
    """
    Exponential distribution random numbers

    Args:
        size (int | tuple[int, ...], optional): shape of the output array. Defaults to 1. Positive integer or tuple of integers.
        scale (real, optional): exponential distribution parameter, mean of the distribution. Defaults to 1.0. Positive real number.

    Returns:
        float | np.ndarray: exponential random numbers
    """
    scale = _ensure_real(scale, "scale")
    if scale <= 0:
        raise ValueError(f"Invalid scale {scale}, expected positive real number")

    return _generate_random_values(size, _core.exp_rand, _core.exp_rands, (scale,))


def uniform(
    size: int | tuple[int, ...] = 1,
    low: real = 0.0,
    high: real = 1.0,
    end: bool = False,
    dtype: DType = DType.Float,
) -> real | np.ndarray:
    """Uniform distribution random numbers

    Args:
        size (int | tuple[int, ...], optional): shape of the output array. Defaults to 1. Positive integer or tuple of integers.
        low (real, optional): lower bound. Defaults to 0.0.
        high (real, optional): upper bound. Defaults to 1.0.
        end (bool, optional): whether to include the upper bound. Defaults to False.
        dtype (DType, optional): data type. Defaults to DType.FLOAT.

    Returns:
        real | np.ndarray: uniform random numbers
    """
    if not isinstance(end, bool):
        raise TypeError(f"end must be a boolean, got {type(end).__name__}")
    if dtype == DType.Float:
        _low = _ensure_real(low, "low")
        _high = _ensure_real(high, "high")
        if _low >= _high:
            raise ValueError("Invalid bounds, low must be less than high")
        return _generate_random_values(
            size,
            _core.uniform_rand_float,
            _core.uniform_rands_float,
            (_low, _high, end),
        )
    elif dtype == DType.Int:
        _low = _ensure_integer(low, "low")
        _high = _ensure_integer(high, "high")
        if _low >= _high:
            raise ValueError("Invalid bounds, low must be less than high")
        return _generate_random_values(
            size, _core.uniform_rand_int, _core.uniform_rands_int, (_low, _high, end)
        )
    else:
        raise ValueError(f"Invalid dtype {dtype}, expected DType.Float or DType.Int")


def randn(
    size: int | tuple[int, ...] = 1, mu: real = 0.0, sigma: real = 1.0
) -> float | np.ndarray:
    """Normal distribution random numbers

    Args:
        size (int | tuple[int, ...], optional): shape of the output array. Defaults to 1. Positive integer or tuple of integers.
        mu (real, optional): mean. Defaults to 0.0.
        sigma (real, optional): standard deviation. Defaults to 1.0. Positive real number.

    Returns:
        float | np.ndarray: normal random numbers
    """
    _mu = _ensure_real(mu, "mu")
    _sigma = _ensure_real(sigma, "sigma")
    if _sigma <= 0:
        raise ValueError(f"Invalid sigma {sigma}, expected positive real number")

    return _generate_random_values(
        size, _core.normal_rand, _core.normal_rands, (_mu, _sigma)
    )


def poisson(size: int | tuple[int, ...] = 1, lambda_: real = 1.0) -> real | np.ndarray:
    """Poisson distribution random numbers

    Args:
        size (int | tuple[int, ...], optional): shape of the output array. Defaults to 1. Positive integer or tuple of integers.
        lambda_ (real, optional): Poisson distribution parameter. Defaults to 1.0. Positive real number.

    Returns:
        real | np.ndarray: Poisson random numbers
    """
    _lambda = _ensure_real(lambda_, "lambda_")
    if _lambda <= 0:
        raise ValueError(f"Invalid lambda {lambda_}, expected positive real number")

    return _generate_random_values(
        size, _core.poisson_rand, _core.poisson_rands, (_lambda,)
    )


def stable_rand(
    alpha: real,
    beta: real,
    sigma: real = 1.0,
    mu: real = 0.0,
    size: int | tuple[int, ...] = 1,
) -> real | np.ndarray:
    """Stable distribution random numbers

    Args:
        alpha (real): stability index. Positive real number, between 0 and 2.
        beta (real): skewness parameter. Real number, between -1 and 1.
        sigma (real, optional): scale parameter. Defaults to 1.0. Positive real number.
        mu (real, optional): location parameter. Defaults to 0.0.
        size (int | tuple[int, ...], optional): shape of the output array. Defaults to 1. Positive integer or tuple of integers.
    Returns:
        real | np.ndarray: stable random numbers
    """
    _alpha = _ensure_real(alpha, "alpha")
    _beta = _ensure_real(beta, "beta")
    _sigma = _ensure_real(sigma, "sigma")
    _mu = _ensure_real(mu, "mu")
    if not (0 < _alpha <= 2):
        raise ValueError(
            f"Invalid alpha {alpha}, expected positive real number between 0 (exclusive) and 2 (inclusive)"
        )
    if not (-1 <= _beta <= 1):
        raise ValueError(f"Invalid beta {beta}, expected real number between -1 and 1")
    if _sigma <= 0:
        raise ValueError(f"Invalid sigma {sigma}, expected positive real number")

    return _generate_random_values(
        size, _core.stable_rand, _core.stable_rands, (_alpha, _beta, _sigma, _mu)
    )


def skew_stable_rand(alpha: real, size: int | tuple[int, ...] = 1) -> real | np.ndarray:
    """Skewed stable distribution random numbers

    Args:
        alpha (real): stability index. Positive real number, between 0 and 1 for S_alpha(1,1,0).
        size (int | tuple[int, ...], optional): shape of the output array. Defaults to 1. Positive integer or tuple of integers.

    Returns:
        real | np.ndarray: skewed stable random numbers
    """
    _alpha = _ensure_real(alpha, "alpha")
    if not (0 < _alpha <= 2):
        raise ValueError(
            f"Invalid alpha {alpha}, expected positive real number, typically (0,2]"
        )

    return _generate_random_values(
        size, _core.skew_stable_rand, _core.skew_stable_rands, (_alpha,)
    )


def bool_rand(size: tuple[int, ...] | int = 1, p: real = 0.5) -> bool | np.ndarray:
    """Boolean random numbers (Bernoulli distribution)

    Args:
        size (tuple[int, ...] | int, optional): shape of the output array. Defaults to 1. Positive integer or tuple of integers.
        p (real, optional): probability of True. Defaults to 0.5. Must be between 0 and 1.

    Returns:
        bool | np.ndarray: boolean random numbers
    """
    _p = _ensure_real(p, "p")
    if not (0 <= _p <= 1):
        raise ValueError(f"Invalid p {p}, probability must be between 0 and 1")

    return _generate_random_values(size, _core.bool_rand, _core.bool_rands, (_p,))
