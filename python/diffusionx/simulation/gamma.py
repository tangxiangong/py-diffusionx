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


class Gamma(ContinuousProcess):
    def __init__(
        self,
        shape: real,  # Also known as alpha or k
        rate: real,  # Also known as beta or 1/theta (if theta is scale)
        start_position: real = 0.0,
    ):
        """
        Initialize a Gamma process object.
        A Gamma process is a Lévy process with non-decreasing sample paths.
        It is characterized by a shape (or concentration) and a rate parameter.

        Args:
            shape (real): Shape parameter (k > 0).
            rate (real): Rate parameter (beta > 0). The scale parameter is 1/rate.
            start_position (real, optional): Starting position. Defaults to 0.0.
                                           Note: Gamma process typically starts at 0 and is non-decreasing.
                                           A non-zero start_position implies X(t) = start_position + G(t),
                                           where G(t) is a standard Gamma process starting at 0.

        Raises:
            TypeError: If shape, rate, or start_position are not numbers.
            ValueError: If shape or rate are not positive.
        """
        try:
            _shape = ensure_float(shape)
            _rate = ensure_float(rate)
            _start_position = ensure_float(start_position)
        except TypeError as e:
            raise TypeError(f"Input parameters must be numbers. Error: {e}") from e

        if _shape <= 0:
            raise ValueError("shape parameter (k) must be positive")
        if _rate <= 0:
            raise ValueError("rate parameter (beta) must be positive")

        self.shape = _shape
        self.rate = _rate
        self.start_position = _start_position

    def simulate(
        self, duration: real, step_size: float = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.gamma_simulate(
            self.start_position,  # Transmitted to _core, which should handle it.
            self.shape,
            self.rate,
            _duration,
            _step_size,
        )

    def moment(
        self,
        duration: real,
        order: int,
        central: bool = False,
        particles: int = 10_000,
        step_size: float = 0.01,
    ) -> float:
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(
            step_size, "step_size"
        )  # Used by _core

        if not isinstance(central, bool):
            raise TypeError("central must be a boolean")

        if _order == 0:
            return 1.0

        result = (
            _core.gamma_raw_moment(
                self.start_position,
                self.shape,
                self.rate,
                _duration,
                _step_size,
                _order,
                _particles,
            )
            if not central
            else _core.gamma_central_moment(
                self.start_position,
                self.shape,
                self.rate,
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
        # For Gamma process, domain[0] is typically current value, domain[1] is target threshold.
        # Since it's non-decreasing, domain[0] < domain[1] is essential.
        _a, _b = validate_domain(domain, process_name="Gamma FPT")
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        # Pass start_position from self, _core.gamma_fpt expects it.
        # The domain (a,b) is relative to the process value space.
        return _core.gamma_fpt(
            self.start_position,  # This is the initial value for the FPT problem
            self.shape,
            self.rate,
            _step_size,
            (_a, _b),  # Target domain for the process value X(t)
            _max_duration,
        )

    def fpt_moment(
        self,
        domain: tuple[real, real],
        order: int,
        central: bool = False,
        particles: int = 10_000,
        step_size: float = 0.01,
        max_duration: real = 1000,
    ) -> Optional[float]:
        _a, _b = validate_domain(domain, process_name="Gamma FPT raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        if not isinstance(central, bool):
            raise TypeError("central must be a boolean")

        result = (
            _core.gamma_fpt_raw_moment(
                self.start_position,
                self.shape,
                self.rate,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _max_duration,
            )
            if not central
            else _core.gamma_fpt_central_moment(
                self.start_position,
                self.shape,
                self.rate,
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
        _a, _b = validate_domain(domain, process_name="Gamma Occupation Time")
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.gamma_occupation_time(
            self.start_position,
            self.shape,
            self.rate,
            _step_size,
            (_a, _b),
            _duration,
        )

    def occupation_time_moment(
        self,
        domain: tuple[real, real],
        duration: real,
        order: int,
        central: bool = False,
        particles: int = 10_000,
        step_size: float = 0.01,
    ) -> float:
        _a, _b = validate_domain(domain, process_name="Gamma Occupation raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if not isinstance(central, bool):
            raise TypeError("central must be a boolean")

        if _order == 0:
            return 1.0

        result = (
            _core.gamma_occupation_time_raw_moment(
                self.start_position,
                self.shape,
                self.rate,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _duration,
            )
            if not central
            else _core.gamma_occupation_time_central_moment(
                self.start_position,
                self.shape,
                self.rate,
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

        return _core.gamma_tamsd(
            self.start_position,
            self.shape,
            self.rate,
            _duration,
            _delta,
            _step_size,
            quad_order,
        )

    def eatamsd(
        self,
        duration: real,
        delta: real,
        particles: int = 10_000,
        step_size: float = 0.01,
        quad_order: int = 10,
    ) -> float:
        _duration = validate_positive_float_param(duration, "duration")
        _delta = validate_positive_float_param(delta, "delta")
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        if not isinstance(quad_order, int) or quad_order <= 0:
            raise ValueError("quad_order must be a positive integer.")

        return _core.gamma_eatamsd(
            self.start_position,
            self.shape,
            self.rate,
            _duration,
            _delta,
            _particles,
            _step_size,
            quad_order,
        )
