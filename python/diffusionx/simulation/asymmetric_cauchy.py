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


class AsymmetricCauchy(StochasticProcess):
    def __init__(
        self,
        start_position: real = 0.0,
        scale: real = 1.0,
        beta: real = 0.0,  # Skewness parameter
    ):
        """
        Initialize an Asymmetric Cauchy process object.

        Args:
            start_position (real, optional): Starting position. Defaults to 0.0.
            scale (real, optional): Scale parameter (gamma > 0). Defaults to 1.0.
            beta (real, optional): Skewness parameter. Must be in [-1, 1]. Defaults to 0.0 (symmetric Cauchy).

        Raises:
            TypeError: If start_position, scale, or beta are not numbers.
            ValueError: If scale is not positive or beta is not in [-1, 1].
        """
        try:
            _start_position = ensure_float(start_position)
            _scale = ensure_float(scale)
            _beta = ensure_float(beta)
        except TypeError as e:
            raise TypeError(f"Input parameters must be numbers. Error: {e}") from e

        if _scale <= 0:
            raise ValueError("scale parameter must be positive")
        if not (-1 <= _beta <= 1):
            raise ValueError(
                f"beta (skewness) must be in the range [-1, 1], got {_beta}"
            )

        self.start_position = _start_position
        self.scale = _scale
        self.beta = _beta

    def __call__(self, duration: real) -> Trajectory:
        return Trajectory(self, duration)

    def simulate(
        self, duration: real, step_size: real = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.asymmetric_cauchy_simulate(
            self.start_position,
            self.scale,
            self.beta,
            _duration,
            _step_size,
        )

    def raw_moment(
        self, duration: real, order: int, particles: int, step_size: real = 0.01
    ) -> float:
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0
        # Higher-order moments are generally infinite or undefined.
        # _core function performs empirical calculation.
        return _core.asymmetric_cauchy_raw_moment(
            self.start_position,
            self.scale,
            self.beta,
            _duration,
            _step_size,
            _order,
            _particles,
        )

    def central_moment(
        self, duration: real, order: int, particles: int, step_size: real = 0.01
    ) -> float:
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0
        # Mean undefined, variance infinite. _core provides empirical.
        return _core.asymmetric_cauchy_central_moment(
            self.start_position,
            self.scale,
            self.beta,
            _duration,
            _step_size,
            _order,
            _particles,
        )

    def fpt(
        self,
        domain: tuple[real, real],
        step_size: real = 0.01,
        max_duration: real = 1000,
    ) -> Optional[float]:
        _a, _b = validate_domain(domain, process_name="AsymmetricCauchy FPT")
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        return _core.asymmetric_cauchy_fpt(
            self.start_position,
            self.scale,
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
        _a, _b = validate_domain(domain, process_name="AsymmetricCauchy FPT raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        return _core.asymmetric_cauchy_fpt_raw_moment(
            self.start_position,
            self.scale,
            self.beta,
            (_a, _b),
            _order,
            _particles,
            _step_size,
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
            domain, process_name="AsymmetricCauchy FPT central moment"
        )
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        if _order == 0:
            return 1.0

        return _core.asymmetric_cauchy_fpt_central_moment(
            self.start_position,
            self.scale,
            self.beta,
            (_a, _b),
            _order,
            _particles,
            _step_size,
            _max_duration,
        )

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
        step_size: real = 0.01,
    ) -> float:
        _a, _b = validate_domain(
            domain, process_name="AsymmetricCauchy Occupation Time"
        )
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.asymmetric_cauchy_occupation_time(
            self.start_position,
            self.scale,
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
            domain, process_name="AsymmetricCauchy Occupation raw moment"
        )
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0

        return _core.asymmetric_cauchy_occupation_time_raw_moment(
            self.start_position,
            self.scale,
            self.beta,
            (_a, _b),
            _order,
            _particles,
            _step_size,
            _duration,
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
            domain, process_name="AsymmetricCauchy Occupation central moment"
        )
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0
        # if _order == 1: return 0.0 # _core provides empirical

        return _core.asymmetric_cauchy_occupation_time_central_moment(
            self.start_position,
            self.scale,
            self.beta,
            (_a, _b),
            _order,
            _particles,
            _step_size,
            _duration,
        )

    def tamsd(
        self,
        duration: real,
        delta: real,
        step_size: real = 0.01,
        quad_order: int = 32,
    ) -> float:
        _duration = validate_positive_float_param(duration, "duration")
        _delta = validate_positive_float_param(delta, "delta")
        _step_size = validate_positive_float_param(step_size, "step_size")
        if not isinstance(quad_order, int) or quad_order <= 0:
            raise ValueError("quad_order must be a positive integer.")

        return _core.asymmetric_cauchy_tamsd(
            self.start_position,
            self.scale,
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
        quad_order: int = 32,
    ) -> float:
        _duration = validate_positive_float_param(duration, "duration")
        _delta = validate_positive_float_param(delta, "delta")
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        if not isinstance(quad_order, int) or quad_order <= 0:
            raise ValueError("quad_order must be a positive integer.")

        return _core.asymmetric_cauchy_eatamsd(
            self.start_position,
            self.scale,
            self.beta,
            _duration,
            _delta,
            _particles,
            _step_size,
            quad_order,
        )
