from math import isfinite
from typing import Union

import numpy as np

from . import random
from .types import DType

real = Union[float, int]


def _ensure_real(value: real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


def _is_real(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class Uniform:
    def __init__(
        self,
        low: real = 0.0,
        high: real = 1.0,
        end: bool = False,
        dtype: DType = DType.Float,
    ):
        """Uniform distribution

        Args:
            low (real, optional): lower bound. Defaults to 0.0.
            high (real, optional): upper bound. Defaults to 1.0.
            end (bool, optional): whether to include the upper bound. Defaults to False.
            dtype (DType, optional): data type. Defaults to DType.FLOAT.
        """
        low = _ensure_real(low, "low")
        high = _ensure_real(high, "high")
        if not isinstance(end, bool):
            raise TypeError(f"end must be a boolean, got {type(end).__name__}")
        if low >= high:
            raise ValueError("Invalid bounds, low must be less than high")
        if dtype not in [DType.Float, DType.Int]:
            raise ValueError("Invalid dtype, must be DType.Float or DType.Int")
        self.low = low
        self.high = high
        self.end = end
        self.dtype = dtype

    def sample(self, size: int | tuple[int, ...] = 1) -> real | np.ndarray:
        """Uniform distribution random numbers

        Args:
            size (int | tuple[int, ...], optional): shape of the output array. Defaults to 1. Positive integer or tuple of integers.

        Returns:
            real | np.ndarray: uniform random numbers
        """
        return random.uniform(size, self.low, self.high, self.end, self.dtype)


class Normal:
    def __init__(self, mu: real = 0.0, sigma: real = 1.0):
        """Normal distribution

        Args:
            mu (real, optional): mean. Defaults to 0.0.
            sigma (real, optional): standard deviation. Defaults to 1.0. Positive real number.
        """
        mu = _ensure_real(mu, "mu")
        sigma = _ensure_real(sigma, "sigma")
        if sigma <= 0:
            raise ValueError("Invalid sigma, must be positive real number")
        self.mu = mu
        self.sigma = sigma

    def sample(self, size: int | tuple[int, ...] = 1) -> real | np.ndarray:
        """Normal distribution random numbers

        Args:
            size (int | tuple[int, ...], optional): shape of the output array. Defaults to 1. Positive integer or tuple of integers.

        Returns:
            real | np.ndarray: normal random numbers
        """
        return random.randn(size, self.mu, self.sigma)

    def __neg__(self):
        return Normal(-self.mu, self.sigma)

    def __add__(self, other):
        if _is_real(other):
            return Normal(self.mu + other, self.sigma)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __radd__(self, other):
        if _is_real(other):
            return Normal(other + self.mu, self.sigma)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for +: '{type(other).__name__}' and '{type(self).__name__}'"
            )

    def __mul__(self, other):
        if _is_real(other):
            return Normal(self.mu * other, self.sigma * abs(other))
        else:
            raise TypeError(
                f"Unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __rmul__(self, other):
        if _is_real(other):
            return Normal(other * self.mu, abs(other) * self.sigma)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for *: '{type(other).__name__}' and '{type(self).__name__}'"
            )


class Exponential:
    def __init__(self, scale: real = 1.0):
        """Exponential distribution

        Args:
            scale (real, optional): scale parameter. Defaults to 1.0. Positive real number.
        """
        scale = _ensure_real(scale, "scale")
        if scale <= 0:
            raise ValueError("Invalid scale, must be positive real number")
        self.scale = scale

    def sample(self, size: int | tuple[int, ...] = 1) -> real | np.ndarray:
        """Exponential distribution random numbers

        Args:
            size (int | tuple[int, ...], optional): shape of the output array. Defaults to 1. Positive integer or tuple of integers.

        Returns:
            real | np.ndarray: exponential random numbers
        """
        return random.randexp(size, self.scale)


class Poisson:
    def __init__(self, lambda_: real = 1.0):
        """Poisson distribution

        Args:
            lambda_ (real, optional): Poisson distribution parameter, mean of the distribution. Defaults to 1.0. Positive real number.
        """
        lambda_ = _ensure_real(lambda_, "lambda_")
        if lambda_ <= 0:
            raise ValueError("Invalid lambda, must be positive real number")
        self.lambda_ = lambda_

    def sample(self, size: int | tuple[int, ...] = 1) -> real | np.ndarray:
        """Poisson distribution random numbers

        Args:
            size (int | tuple[int, ...], optional): shape of the output array. Defaults to 1. Positive integer or tuple of integers.

        Returns:
            real | np.ndarray: Poisson distribution random numbers
        """
        return random.poisson(size, self.lambda_)


class Stable:
    def __init__(self, alpha: real, beta: real, sigma: real, mu: real):
        """Stable distribution

        Args:
            alpha (real): stability index. Positive real number, between 0 and 2.
            beta (real): skewness parameter. Real number, between -1 and 1.
            sigma (real): scale parameter. Positive real number.
            mu (real): location parameter. Real number.
        """
        alpha = _ensure_real(alpha, "alpha")
        beta = _ensure_real(beta, "beta")
        sigma = _ensure_real(sigma, "sigma")
        mu = _ensure_real(mu, "mu")
        if alpha <= 0 or alpha > 2:
            raise ValueError(
                "Invalid alpha, must be positive real number between 0 and 2"
            )
        if beta < -1 or beta > 1:
            raise ValueError("Invalid beta, must be real number between -1 and 1")
        if sigma <= 0:
            raise ValueError("Invalid sigma, must be positive real number")

        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.mu = mu
        self.symm: bool = False
        self.skewed: bool = False
        self.std: bool = False

    @classmethod
    def symmetric(cls, alpha: real):
        """Symmetric stable distribution

        Args:
            alpha (real): stability index. Positive real number, between 0 and 2.
        """
        result = cls(alpha, 0.0, 1.0, 0.0)
        result.symm = True
        return result

    @classmethod
    def skew(cls, alpha: real):
        """Skewed stable distribution

        Args:
            alpha (real): stability index. Positive real number, between 0 and 1.
        """
        alpha = _ensure_real(alpha, "alpha")
        if alpha <= 0 or alpha > 1:
            raise ValueError(
                "Invalid alpha, must be positive real number between 0 and 1"
            )
        result = cls(alpha, 1.0, 1.0, 0.0)
        result.skewed = True
        return result

    @classmethod
    def standard(cls, alpha: real, beta: real):
        """Standard stable distribution

        Args:
            alpha (real): stability index. Positive real number, between 0 and 2.
            beta (real): skewness parameter. Real number, between -1 and 1.
        """
        result = cls(alpha, beta, 1.0, 0.0)
        result.std = True
        return result

    def sample(self, size: int | tuple[int, ...] = 1) -> real | np.ndarray:
        """Stable distribution random numbers

        Args:
            size (int | tuple[int, ...], optional): shape of the output array. Defaults to 1. Positive integer or tuple of integers.

        Returns:
            real | np.ndarray: stable random numbers
        """
        if self.skewed:
            return random.skew_stable_rand(self.alpha, size)
        else:
            return random.stable_rand(self.alpha, self.beta, self.sigma, self.mu, size)

    def __neg__(self):
        return Stable(self.alpha, -self.beta, self.sigma, -self.mu)

    def __add__(self, other):
        if _is_real(other):
            return Stable(self.alpha, self.beta, self.sigma, self.mu + other)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __radd__(self, other):
        if _is_real(other):
            return Stable(self.alpha, self.beta, self.sigma, other + self.mu)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for +: '{type(other).__name__}' and '{type(self).__name__}'"
            )

    def __mul__(self, other):
        if _is_real(other):
            # When multiplying a stable distribution by a scalar c,
            # new_sigma = sigma * |c|
            # new_mu = mu * c if alpha != 1
            # new_mu = mu * c + sigma * c * beta * (2/pi) * ln|c| if alpha == 1.
            # Current implementation simplifies for mu, assuming alpha != 1 or ignoring the second term for alpha = 1.
            new_sigma = self.sigma * abs(other)
            new_mu = self.mu * other
            return Stable(self.alpha, self.beta, new_sigma, new_mu)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __rmul__(self, other):
        if _is_real(other):
            # Similar to __mul__
            new_sigma = abs(other) * self.sigma
            new_mu = other * self.mu
            return Stable(self.alpha, self.beta, new_sigma, new_mu)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for *: '{type(other).__name__}' and '{type(self).__name__}'"
            )


class SkewStable(Stable):
    def __init__(self, alpha: real):
        alpha = _ensure_real(alpha, "alpha")
        if alpha <= 0 or alpha > 1:
            raise ValueError(
                "Invalid alpha, must be positive real number between 0 and 1"
            )
        super().__init__(alpha, 1.0, 1.0, 0.0)
        self.skewed = True

    def sample(self, size: int | tuple[int, ...] = 1) -> real | np.ndarray:
        return random.skew_stable_rand(self.alpha, size)
