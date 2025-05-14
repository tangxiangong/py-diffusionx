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


class Bb(StochasticProcess):
    def __init__(
        self,
        start_value: real = 0.0,
        end_value: real = 0.0,
        diffusion_coefficient: real = 1.0,
    ):
        """
        Initialize a Brownian Bridge (Bb) object.

        Args:
            start_value (real, optional): Starting value of the Brownian Bridge. Defaults to 0.0.
            end_value (real, optional): Ending value of the Brownian Bridge at the specified duration. Defaults to 0.0.
            diffusion_coefficient (real, optional): Diffusion coefficient. Defaults to 1.0.

        Raises:
            TypeError: If start_value, end_value, or diffusion_coefficient are not numbers.
            ValueError: If diffusion_coefficient is not positive.
        """
        try:
            _start_value = ensure_float(start_value)
            _end_value = ensure_float(end_value)
            _diffusion_coefficient = ensure_float(diffusion_coefficient)
        except TypeError as e:
            raise TypeError(f"Input parameters must be numbers. Error: {e}") from e

        if _diffusion_coefficient <= 0:
            raise ValueError("diffusion_coefficient must be positive")

        self.start_value = _start_value
        self.end_value = _end_value
        self.diffusion_coefficient = _diffusion_coefficient

    def __call__(self, duration: real) -> Trajectory:
        return Trajectory(self, duration)

    def simulate(
        self, duration: real, step_size: real = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.bb_simulate(
            self.start_value,
            self.end_value,
            self.diffusion_coefficient,
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

        return _core.bb_raw_moment(
            self.start_value,
            self.end_value,
            self.diffusion_coefficient,
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
        if _order == 1:
            # For a fixed-endpoint process like Brownian Bridge, the mean at any t < duration
            # is interpolated. The first central moment around this interpolated mean is 0.
            # If the _core function calculates this, direct call is fine.
            # Otherwise, this might need specific handling if _core implies moment around 0.
            # Assuming _core.bb_central_moment handles it correctly around the time-t mean.
            pass  # Let the core function handle it, could be non-zero if definition varies.

        return _core.bb_central_moment(
            self.start_value,
            self.end_value,
            self.diffusion_coefficient,
            _duration,
            _step_size,
            _order,
            _particles,
        )

    def fpt(
        self,
        domain: tuple[real, real],
        step_size: real = 0.01,
        max_duration: real = 1000,  # This is effectively the bridge's duration
    ) -> Optional[float]:
        _a, _b = validate_domain(domain, process_name="Bb FPT")
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(
            max_duration, "max_duration (bridge duration)"
        )

        return _core.bb_fpt(
            self.start_value,
            self.end_value,
            self.diffusion_coefficient,
            _step_size,
            (_a, _b),
            _max_duration,  # This should be the fixed duration of the bridge for FPT context
        )

    def fpt_raw_moment(
        self,
        domain: tuple[real, real],
        order: int,
        particles: int,
        step_size: real = 0.01,
        max_duration: real = 1000,  # Bridge duration
    ) -> Optional[float]:
        _a, _b = validate_domain(domain, process_name="Bb FPT raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(
            max_duration, "max_duration (bridge duration)"
        )

        return _core.bb_fpt_raw_moment(
            self.start_value,
            self.end_value,
            self.diffusion_coefficient,
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
        max_duration: real = 1000,  # Bridge duration
    ) -> Optional[float]:
        _a, _b = validate_domain(domain, process_name="Bb FPT central moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(
            max_duration, "max_duration (bridge duration)"
        )

        if _order == 0:
            return 1.0

        return _core.bb_fpt_central_moment(
            self.start_value,
            self.end_value,
            self.diffusion_coefficient,
            (_a, _b),
            _order,
            _particles,
            _step_size,
            _max_duration,
        )

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,  # Bridge duration
        step_size: real = 0.01,
    ) -> float:
        _a, _b = validate_domain(domain, process_name="Bb Occupation Time")
        _duration = validate_positive_float_param(
            duration, "duration (bridge duration)"
        )
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.bb_occupation_time(
            self.start_value,
            self.end_value,
            self.diffusion_coefficient,
            _step_size,
            (_a, _b),
            _duration,
        )

    def occupation_time_raw_moment(
        self,
        domain: tuple[real, real],
        order: int,
        particles: int,
        duration: real,  # Bridge duration
        step_size: real = 0.01,
    ) -> float:
        _a, _b = validate_domain(domain, process_name="Bb Occupation raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(
            duration, "duration (bridge duration)"
        )
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0

        return _core.bb_occupation_time_raw_moment(
            self.start_value,
            self.end_value,
            self.diffusion_coefficient,
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
        duration: real,  # Bridge duration
        step_size: real = 0.01,
    ) -> float:
        _a, _b = validate_domain(domain, process_name="Bb Occupation central moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(
            duration, "duration (bridge duration)"
        )
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0
        # if _order == 1: # For occupation time, first central moment is generally 0.
        #     return 0.0 # Assuming _core handles this.

        return _core.bb_occupation_time_central_moment(
            self.start_value,
            self.end_value,
            self.diffusion_coefficient,
            (_a, _b),
            _order,
            _particles,
            _step_size,
            _duration,
        )

    def tamsd(
        self,
        duration: real,  # Bridge duration
        delta: real,
        step_size: real = 0.01,
        quad_order: int = 32,
    ) -> float:
        _duration = validate_positive_float_param(
            duration, "duration (bridge duration)"
        )
        _delta = validate_positive_float_param(delta, "delta")
        _step_size = validate_positive_float_param(step_size, "step_size")
        if not isinstance(quad_order, int) or quad_order <= 0:
            raise ValueError("quad_order must be a positive integer.")

        return _core.bb_tamsd(
            self.start_value,
            self.end_value,
            self.diffusion_coefficient,
            _duration,
            _delta,
            _step_size,
            quad_order,
        )

    def eatamsd(
        self,
        duration: real,  # Bridge duration
        delta: real,
        particles: int,
        step_size: real = 0.01,
        quad_order: int = 32,
    ) -> float:
        _duration = validate_positive_float_param(
            duration, "duration (bridge duration)"
        )
        _delta = validate_positive_float_param(delta, "delta")
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        if not isinstance(quad_order, int) or quad_order <= 0:
            raise ValueError("quad_order must be a positive integer.")

        return _core.bb_eatamsd(
            self.start_value,
            self.end_value,
            self.diffusion_coefficient,
            _duration,
            _delta,
            _particles,
            _step_size,
            quad_order,
        )
