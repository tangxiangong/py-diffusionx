from diffusionx import _core
from typing import Union, Optional
from .basic import StochasticProcess, Trajectory
from .utils import (
    ensure_float,
    validate_domain,
    validate_order,
    validate_particles,
    validate_positive_float_param,
)
import numpy as np


real = Union[float, int]


class AsymmetricLevy(StochasticProcess):
    def __init__(
        self,
        alpha: real,
        beta: real,
        start_position: real = 0.0,
    ):
        """
        Initialize an Asymmetric Lévy process object.

        Args:
            alpha (real): Stability index of the Asymmetric Lévy process. Must be in (0, 2].
            beta (real): Skewness parameter. Must be in [-1, 1].
            start_position (real, optional): Starting position. Defaults to 0.0.

        Raises:
            TypeError: If alpha, beta, or start_position are not numbers.
            ValueError: If alpha is not in (0, 2] or beta is not in [-1, 1].
        """
        try:
            _alpha = ensure_float(alpha)
            _beta = ensure_float(beta)
            _start_position = ensure_float(start_position)
        except TypeError as e:
            raise TypeError(f"Input parameters must be numbers. Error: {e}") from e

        if not (0 < _alpha <= 2):
            raise ValueError(f"alpha must be in the range (0, 2], got {_alpha}")
        if not (-1 <= _beta <= 1):
            raise ValueError(f"beta must be in the range [-1, 1], got {_beta}")

        self.alpha = _alpha
        self.beta = _beta
        self.start_position = _start_position

    def __call__(self, duration: real) -> Trajectory:
        return Trajectory(self, duration)

    def simulate(
        self, duration: real, step_size: real = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.asymmetric_levy_simulate(
            self.start_position,
            self.alpha,
            self.beta,
            _duration,
            _step_size,
        )

    def fpt(
        self,
        domain: tuple[real, real],
        step_size: real = 0.01,
        max_duration: real = 1000,
    ) -> Optional[float]:
        _a, _b = validate_domain(domain, process_name="AsymmetricLevy FPT")
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        return _core.asymmetric_levy_fpt(
            self.start_position,
            self.alpha,
            self.beta,
            _step_size,
            (_a, _b),
            _max_duration,
        )

    def fpt_raw_moment(
        self,
        domain: tuple[real, real],
        order: int,
        particles: int,
        step_size: real = 0.01,
        max_duration: real = 1000,
    ) -> Optional[float]:
        _a, _b = validate_domain(domain, process_name="AsymmetricLevy FPT raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        return _core.asymmetric_levy_fpt_raw_moment(
            self.start_position,
            self.alpha,
            self.beta,
            (_a, _b),
            _order,
            _particles,
            _step_size,  # Corrected order of arguments
            _max_duration,
        )

    def fpt_central_moment(
        self,
        domain: tuple[real, real],
        order: int,
        particles: int,
        step_size: real = 0.01,
        max_duration: real = 1000,
    ) -> Optional[float]:
        _a, _b = validate_domain(
            domain, process_name="AsymmetricLevy FPT central moment"
        )
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        if _order == 0:
            return 1.0

        return _core.asymmetric_levy_fpt_central_moment(
            self.start_position,
            self.alpha,
            self.beta,
            (_a, _b),
            _order,
            _particles,
            _step_size,  # Corrected order of arguments
            _max_duration,
        )

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
        step_size: real = 0.01,
    ) -> float:
        _a, _b = validate_domain(domain, process_name="AsymmetricLevy Occupation Time")
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.asymmetric_levy_occupation_time(
            self.start_position,
            self.alpha,
            self.beta,
            _step_size,
            (_a, _b),
            _duration,
        )

    def occupation_time_raw_moment(
        self,
        domain: tuple[real, real],
        order: int,
        particles: int,
        duration: real,
        step_size: real = 0.01,
    ) -> float:
        _a, _b = validate_domain(
            domain, process_name="AsymmetricLevy Occupation raw moment"
        )
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0

        return _core.asymmetric_levy_occupation_time_raw_moment(
            self.start_position,
            self.alpha,
            self.beta,
            (_a, _b),
            _duration,  # Corrected order
            _order,
            _particles,
            _step_size,
        )

    def occupation_time_central_moment(
        self,
        domain: tuple[real, real],
        order: int,
        particles: int,
        duration: real,
        step_size: real = 0.01,
    ) -> float:
        _a, _b = validate_domain(
            domain, process_name="AsymmetricLevy Occupation central moment"
        )
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0
        if _order == 1:
            return 0.0

        return _core.asymmetric_levy_occupation_time_central_moment(
            self.start_position,
            self.alpha,
            self.beta,
            (_a, _b),
            _duration,  # Corrected order
            _order,
            _particles,
            _step_size,
        )

    def tamsd(
        self,
        duration: real,
        delta: real,
        step_size: real = 0.01,
        quad_order: int = 32,  # Default from _core.pyi for bm
    ) -> float:
        _duration = validate_positive_float_param(duration, "duration")
        _delta = validate_positive_float_param(delta, "delta")
        _step_size = validate_positive_float_param(step_size, "step_size")
        if not isinstance(quad_order, int) or quad_order <= 0:
            raise ValueError("quad_order must be a positive integer.")

        return _core.asymmetric_levy_tamsd(
            self.start_position,
            self.alpha,
            self.beta,
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
        step_size: real = 0.01,
        quad_order: int = 32,  # Default from _core.pyi for bm
    ) -> float:
        _duration = validate_positive_float_param(duration, "duration")
        _delta = validate_positive_float_param(delta, "delta")
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        if not isinstance(quad_order, int) or quad_order <= 0:
            raise ValueError("quad_order must be a positive integer.")

        return _core.asymmetric_levy_eatamsd(
            self.start_position,
            self.alpha,
            self.beta,
            _duration,
            _delta,
            _particles,
            _step_size,
            quad_order,
        )
