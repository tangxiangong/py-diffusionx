from diffusionx import _core
from typing import Union, Optional
from .basic import ContinuousProcess
from .utils import (
    ensure_float,
    validate_domain,
    validate_order,
    validate_particles,
    validate_positive_float_param,
)
import numpy as np

real = Union[float, int]


class OrnsteinUhlenbeck(ContinuousProcess):
    def __init__(
        self,
        theta: real,  # Mean reversion rate
        mu: real,  # Long-term mean
        sigma: real,  # Volatility
        start_position: real = 0.0,
    ):
        """
        Initialize an Ornstein-Uhlenbeck (OU) process object.
        dX_t = theta * (mu - X_t) dt + sigma * dW_t

        Args:
            theta (real): Mean reversion rate (theta > 0).
            mu (real): Long-term mean (equilibrium level).
            sigma (real): Volatility (sigma > 0).
            start_position (real, optional): Starting position of the process. Defaults to mu.

        Raises:
            TypeError: If theta, mu, sigma, or start_position are not numbers.
            ValueError: If theta or sigma are not positive.
        """
        try:
            _theta = ensure_float(theta)
            _mu = ensure_float(mu)
            _sigma = ensure_float(sigma)
            _start_position = ensure_float(start_position)
        except TypeError as e:
            raise TypeError(f"Input parameters must be numbers. Error: {e}") from e

        if _theta <= 0:
            raise ValueError("theta (mean reversion rate) must be positive")
        if _sigma <= 0:
            raise ValueError("sigma (volatility) must be positive")

        self.theta = _theta
        self.mu = _mu
        self.sigma = _sigma
        self.start_position = _start_position

    def simulate(
        self, duration: real, step_size: float = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.ou_simulate(
            self.start_position,
            self.theta,
            self.mu,
            self.sigma,
            _duration,
            _step_size,
        )

    def moment(
        self,
        duration: real,
        order: int,
        center: bool = False,
        particles: int = 10_000,
        step_size: float = 0.01,
    ) -> float:
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0

        result = (
            _core.ou_raw_moment(
                self.start_position,
                self.theta,
                self.mu,
                self.sigma,
                _duration,
                _step_size,
                _order,
                _particles,
            )
            if not center
            else _core.ou_central_moment(
                self.start_position,
                self.theta,
                self.mu,
                self.sigma,
                _duration,
                _step_size,
                _order,
                _particles,
            )
        )

        return result

    def fpt(
        self,
        domain: tuple[real, real],
        step_size: float = 0.01,
        max_duration: real = 1000,
    ) -> Optional[float]:
        _a, _b = validate_domain(domain, process_name="Ou FPT")
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        return _core.ou_fpt(
            self.start_position,
            self.theta,
            self.mu,
            self.sigma,
            _step_size,
            (_a, _b),
            _max_duration,
        )

    def fpt_moment(
        self,
        domain: tuple[real, real],
        order: int,
        center: bool = False,
        particles: int = 10_000,
        step_size: float = 0.01,
        max_duration: real = 1000,
    ) -> Optional[float]:
        _a, _b = validate_domain(domain, process_name="Ou FPT raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        result = (
            _core.ou_fpt_raw_moment(
                self.start_position,
                self.theta,
                self.mu,
                self.sigma,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _max_duration,
            )
            if not center
            else _core.ou_fpt_central_moment(
                self.start_position,
                self.theta,
                self.mu,
                self.sigma,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _max_duration,
            )
        )

        return result

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
        step_size: float = 0.01,
    ) -> float:
        _a, _b = validate_domain(domain, process_name="Ou Occupation Time")
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.ou_occupation_time(
            self.start_position,
            self.theta,
            self.mu,
            self.sigma,
            _step_size,
            (_a, _b),
            _duration,
        )

    def occupation_time_moment(
        self,
        domain: tuple[real, real],
        duration: real,
        order: int,
        center: bool = False,
        particles: int = 10_000,
        step_size: float = 0.01,
    ) -> float:
        _a, _b = validate_domain(domain, process_name="Ou Occupation raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0

        result = (
            _core.ou_occupation_time_raw_moment(
                self.start_position,
                self.theta,
                self.mu,
                self.sigma,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _duration,
            )
            if not center
            else _core.ou_occupation_time_central_moment(
                self.start_position,
                self.theta,
                self.mu,
                self.sigma,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _duration,
            )
        )

        return result

    def tamsd(
        self,
        duration: real,
        delta: real,
        step_size: float = 0.01,
        quad_order: int = 10,
    ) -> float:
        _duration = validate_positive_float_param(duration, "duration")
        _delta = validate_positive_float_param(delta, "delta")
        _step_size = validate_positive_float_param(step_size, "step_size")
        if not isinstance(quad_order, int) or quad_order <= 0:
            raise ValueError("quad_order must be a positive integer.")

        return _core.ou_tamsd(
            self.start_position,
            self.theta,
            self.mu,
            self.sigma,
            _duration,
            _delta,
            _step_size,
            quad_order,
        )

    def eatamsd(
        self,
        duration: real,
        delta: real,
        particles: int,
        step_size: float = 0.01,
        quad_order: int = 10,
    ) -> float:
        _duration = validate_positive_float_param(duration, "duration")
        _delta = validate_positive_float_param(delta, "delta")
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        if not isinstance(quad_order, int) or quad_order <= 0:
            raise ValueError("quad_order must be a positive integer.")

        return _core.ou_eatamsd(
            self.start_position,
            self.theta,
            self.mu,
            self.sigma,
            _duration,
            _delta,
            _particles,
            _step_size,
            quad_order,
        )
