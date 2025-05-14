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


class Cauchy(StochasticProcess):
    def __init__(
        self,
        start_position: real = 0.0,
        scale: real = 1.0,  # Corresponds to gamma in some parametrizations
    ):
        """
        Initialize a Cauchy process object.

        Args:
            start_position (real, optional): Starting position. Defaults to 0.0.
            scale (real, optional): Scale parameter (gamma > 0). Defaults to 1.0.

        Raises:
            TypeError: If start_position or scale are not numbers.
            ValueError: If scale is not positive.
        """
        try:
            _start_position = ensure_float(start_position)
            _scale = ensure_float(scale)
        except TypeError as e:
            raise TypeError(f"Input parameters must be numbers. Error: {e}") from e

        if _scale <= 0:
            raise ValueError("scale parameter must be positive")

        self.start_position = _start_position
        self.scale = _scale

    def __call__(self, duration: real) -> Trajectory:
        return Trajectory(self, duration)

    def simulate(
        self,
        duration: real,
        step_size: real = 0.01,  # step_size for Cauchy is more like a dt for discretization
    ) -> tuple[np.ndarray, np.ndarray]:
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.cauchy_simulate(
            self.start_position,
            self.scale,
            _duration,
            _step_size,
        )

    # Cauchy process moments (raw/central other than 0th) are typically undefined or infinite.
    # The _core functions might simulate them empirically if requested, but it's often not meaningful.
    # We will provide wrappers if they exist in _core, assuming empirical calculation.

    def raw_moment(
        self, duration: real, order: int, particles: int, step_size: real = 0.01
    ) -> float:
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0
        # For order > 0, moments are generally infinite for Cauchy.
        # If _core.cauchy_raw_moment exists, it's an empirical estimate.
        if hasattr(_core, "cauchy_raw_moment"):
            return _core.cauchy_raw_moment(
                self.start_position,
                self.scale,
                _duration,
                _step_size,
                _order,
                _particles,
            )
        else:
            # Or raise NotImplementedError, or return float('inf') if appropriate.
            # For now, let's assume if not in _core, it was an oversight or not to be called.
            # Check _core.pyi again. It IS defined there.
            return _core.cauchy_raw_moment(
                self.start_position,
                self.scale,
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
        # Mean (order 1 central moment) is undefined. Variance (order 2) is infinite.
        # Again, relying on empirical simulation from _core if it exists.
        if hasattr(_core, "cauchy_central_moment"):
            if _order == 1:  # Median is start_position, but mean is undefined.
                # The _core function would return an empirical mean.
                pass
            return _core.cauchy_central_moment(
                self.start_position,
                self.scale,
                _duration,
                _step_size,
                _order,
                _particles,
            )
        else:
            # Similar to raw_moment
            return _core.cauchy_central_moment(
                self.start_position,
                self.scale,
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
        _a, _b = validate_domain(domain, process_name="Cauchy FPT")
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        return _core.cauchy_fpt(
            self.start_position,
            self.scale,
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
        _a, _b = validate_domain(domain, process_name="Cauchy FPT raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        return _core.cauchy_fpt_raw_moment(
            self.start_position,
            self.scale,
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
        _a, _b = validate_domain(domain, process_name="Cauchy FPT central moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        if _order == 0:
            return 1.0

        return _core.cauchy_fpt_central_moment(
            self.start_position,
            self.scale,
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
        _a, _b = validate_domain(domain, process_name="Cauchy Occupation Time")
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.cauchy_occupation_time(
            self.start_position,
            self.scale,
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
        _a, _b = validate_domain(domain, process_name="Cauchy Occupation raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0

        return _core.cauchy_occupation_time_raw_moment(
            self.start_position,
            self.scale,
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
            domain, process_name="Cauchy Occupation central moment"
        )
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0
        # if _order == 1: return 0.0 # For occupation time of symmetric process, maybe.

        return _core.cauchy_occupation_time_central_moment(
            self.start_position,
            self.scale,
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
        # TAMSD for Cauchy process is usually infinite or non-convergent due to undefined variance.
        # If _core provides it, it's an empirical calculation.
        _duration = validate_positive_float_param(duration, "duration")
        _delta = validate_positive_float_param(delta, "delta")
        _step_size = validate_positive_float_param(step_size, "step_size")
        if not isinstance(quad_order, int) or quad_order <= 0:
            raise ValueError("quad_order must be a positive integer.")

        return _core.cauchy_tamsd(
            self.start_position,
            self.scale,
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

        return _core.cauchy_eatamsd(
            self.start_position,
            self.scale,
            _duration,
            _delta,
            _particles,
            _step_size,
            quad_order,
        )
