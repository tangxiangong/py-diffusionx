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


class Levy(ContinuousProcess):
    def __init__(
        self,
        alpha: real,
        start_position: real = 0.0,
    ):
        """
        Initialize a Lévy process object.

        Args:
            alpha (real): Stability index of the Lévy process. Must be in the range (0, 2].
            start_position (real, optional): Starting position of the Lévy process. Defaults to 0.0.

        Raises:
            TypeError: If alpha or start_position are not numbers.
            ValueError: If alpha is not in the range (0, 2].
        """
        try:
            _start_position = ensure_float(start_position)
            _alpha = ensure_float(alpha)
        except TypeError as e:
            raise TypeError(
                f"Input parameters alpha and start_position must be numbers. Error: {e}"
            ) from e

        if not (0 < _alpha <= 2):
            raise ValueError(f"alpha must be in the range (0, 2], got {_alpha}")

        self.start_position = _start_position
        self.alpha = _alpha

    def simulate(
        self, duration: real, step_size: float = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the Lévy process.

        Args:
            duration (real): Total duration of the simulation.
            step_size (real, optional): Step size for the simulation. Defaults to 0.01.

        Raises:
            TypeError: If duration or step_size are not numbers.
            ValueError: If duration or step_size are not positive.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the times and positions of the Lévy process.
        """
        try:
            _duration = ensure_float(duration)
            _step_size = ensure_float(step_size)
        except TypeError as e:
            raise TypeError(
                f"duration and step_size must be numbers. Error: {e}"
            ) from e

        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _step_size <= 0:
            raise ValueError("step_size must be positive")

        return _core.levy_simulate(
            self.start_position,
            self.alpha,
            _duration,
            _step_size,
        )

    def fpt(
        self,
        domain: tuple[real, real],
        step_size: float = 0.01,
        max_duration: real = 1000,
    ) -> Optional[float]:
        """
        Calculate the first passage time of the Lévy process.

        Args:
            domain (tuple[real, real]): The domain (a, b) for FPT. a must be less than b.
            step_size (real, optional): Step size for the simulation. Defaults to 0.01.
            max_duration (real, optional): Maximum duration to simulate for FPT. Defaults to 1000.

        Raises:
            TypeError: If domain elements, step_size, or max_duration are not numbers.
            ValueError: If domain is not a valid interval (a >= b), or if step_size or max_duration are not positive.

        Returns:
            Optional[float]: The first passage time, or None if max_duration is reached before FPT.
        """
        _a, _b = validate_domain(domain, process_name="Levy FPT")
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        return _core.levy_fpt(
            self.start_position,
            self.alpha,
            _step_size,
            (_a, _b),
            _max_duration,
        )

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
        step_size: float = 0.01,
    ) -> float:
        """
        Calculate the occupation time of the Lévy process in a given domain.

        Args:
            domain (tuple[real, real]): The domain (a, b) for occupation time. a must be less than b.
            duration (real): The total duration of the simulation.
            step_size (real, optional): Step size for the simulation. Defaults to 0.01.

        Raises:
            TypeError: If domain elements, duration, or step_size are not numbers.
            ValueError: If domain is not a valid interval (a >= b), or if duration or step_size are not positive.

        Returns:
            float: The occupation time of the Lévy process in the domain.
        """
        _a, _b = validate_domain(domain, process_name="Levy Occupation Time")
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.levy_occupation_time(
            self.start_position,
            self.alpha,
            _step_size,
            (_a, _b),
            _duration,
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
        _a, _b = validate_domain(domain, process_name="Levy FPT raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        result = (
            _core.levy_fpt_raw_moment(
                self.start_position,
                self.alpha,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _max_duration,
            )
            if not center
            else _core.levy_fpt_central_moment(
                self.start_position,
                self.alpha,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _max_duration,
            )
        )

        return result

    def occupation_time_moment(
        self,
        domain: tuple[real, real],
        duration: real,
        order: int,
        center: bool = False,
        particles: int = 10_000,
        step_size: float = 0.01,
    ) -> float:
        _a, _b = validate_domain(domain, process_name="Levy Occupation raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0

        result = (
            _core.levy_occupation_time_raw_moment(
                self.start_position,
                self.alpha,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _duration,
            )
            if not center
            else _core.levy_occupation_time_central_moment(
                self.start_position,
                self.alpha,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _duration,
            )
        )

        return result


class Subordinator(ContinuousProcess):
    def __init__(
        self,
        alpha: real,
    ):
        """
        Initialize a subordinator object.

        Args:
            alpha (real): The alpha parameter of the subordinator, the value must be in the range (0, 1).

        Raises:
            ValueError: If alpha is not in the range (0, 1).

        Returns:
            Subordinator: A subordinator object.
        """
        alpha_transformed = ensure_float(alpha)
        if alpha_transformed <= 0 or alpha_transformed >= 1:
            raise ValueError("alpha must be in the range (0, 1) for Subordinator")

        self.alpha = alpha_transformed

    def simulate(
        self, duration: real, step_size: float = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        _duration = ensure_float(duration)
        _step_size = ensure_float(step_size)
        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _step_size <= 0:
            raise ValueError("step_size must be positive")
        return _core.subordinator_simulate(
            self.alpha,
            _duration,
            _step_size,
        )

    def fpt(
        self,
        domain: tuple[real, real],
        step_size: float = 0.01,
        max_duration: real = 1000,
    ) -> Optional[float]:
        _a, _b = validate_domain(domain, process_name="Subordinator FPT")
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        return _core.subordinator_fpt(
            self.alpha,
            (_a, _b),
            _max_duration,
            _step_size,
        )

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
        step_size: float = 0.01,
    ) -> float:
        _a, _b = validate_domain(domain, process_name="Subordinator Occupation Time")
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")
        # Original code: a >= b -> "domain must be a valid interval"
        return _core.subordinator_occupation_time(
            self.alpha,
            (_a, _b),
            _duration,
            _step_size,
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
        _a, _b = validate_domain(domain, process_name="Subordinator FPT raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        result = (
            _core.subordinator_fpt_raw_moment(
                self.alpha,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _max_duration,
            )
            if not center
            else _core.subordinator_fpt_central_moment(
                self.alpha,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _max_duration,
            )
        )

        return result

    def occupation_time_moment(
        self,
        domain: tuple[real, real],
        duration: real,
        order: int,
        center: bool = False,
        particles: int = 10_000,
        step_size: float = 0.01,
    ) -> float:
        _a, _b = validate_domain(
            domain, process_name="Subordinator Occupation raw moment"
        )
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0

        result = (
            _core.subordinator_occupation_time_raw_moment(
                self.alpha,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _duration,
            )
            if not center
            else _core.subordinator_occupation_time_central_moment(
                self.alpha,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _duration,
            )
        )

        return result


class InvSubordinator(ContinuousProcess):
    def __init__(
        self,
        alpha: real,
    ):
        """
        Initialize an inverse subordinator object.

        Args:
            alpha (real): The alpha parameter of the inverse subordinator, value must be in (0, 1).
        """
        alpha_transformed = ensure_float(alpha)
        if alpha_transformed <= 0 or alpha_transformed >= 1:
            raise ValueError("alpha must be in the range (0, 1) for InvSubordinator")
        self.alpha = alpha_transformed

    def simulate(
        self, duration: real, step_size: float = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        _duration = ensure_float(duration)
        _step_size = ensure_float(step_size)
        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _step_size <= 0:
            raise ValueError("step_size must be positive")
        return _core.inv_subordinator_simulate(
            self.alpha,
            _duration,
            _step_size,
        )

    def fpt(
        self,
        domain: tuple[real, real],
        step_size: float = 0.01,
        max_duration: real = 1000,
    ) -> Optional[float]:
        _a, _b = validate_domain(domain, process_name="InvSubordinator FPT")
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        return _core.inv_subordinator_fpt(
            self.alpha,
            (_a, _b),
            _max_duration,
            _step_size,
        )

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
        step_size: float = 0.01,
    ) -> float:
        _a, _b = validate_domain(domain, process_name="InvSubordinator Occupation Time")
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        return _core.inv_subordinator_occupation_time(
            self.alpha,
            (_a, _b),
            _duration,
            _step_size,
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
        _a, _b = validate_domain(domain, process_name="InvSubordinator FPT raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        result = (
            _core.inv_subordinator_fpt_raw_moment(
                self.alpha,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _max_duration,
            )
            if not center
            else _core.inv_subordinator_fpt_central_moment(
                self.alpha,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _max_duration,
            )
        )

        return result

    def occupation_time_moment(
        self,
        domain: tuple[real, real],
        duration: real,
        order: int,
        center: bool = False,
        particles: int = 10_000,
        step_size: float = 0.01,
    ) -> float:
        _a, _b = validate_domain(
            domain, process_name="InvSubordinator Occupation raw moment"
        )
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _duration = validate_positive_float_param(duration, "duration")
        _step_size = validate_positive_float_param(step_size, "step_size")

        if _order == 0:
            return 1.0

        result = (
            _core.inv_subordinator_occupation_time_raw_moment(
                self.alpha,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _duration,
            )
            if not center
            else _core.inv_subordinator_occupation_time_central_moment(
                self.alpha,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _duration,
            )
        )

        return result


class AsymmetricLevy(ContinuousProcess):
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

    def simulate(
        self, duration: real, step_size: float = 0.01
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
        step_size: float = 0.01,
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

    def fpt_moment(
        self,
        domain: tuple[real, real],
        order: int,
        center: bool = False,
        particles: int = 10_000,
        step_size: float = 0.01,
        max_duration: real = 1000,
    ) -> Optional[float]:
        _a, _b = validate_domain(domain, process_name="AsymmetricLevy FPT raw moment")
        _order = validate_order(order)
        _particles = validate_particles(particles)
        _step_size = validate_positive_float_param(step_size, "step_size")
        _max_duration = validate_positive_float_param(max_duration, "max_duration")

        result = (
            _core.asymmetric_levy_fpt_raw_moment(
                self.start_position,
                self.alpha,
                self.beta,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _max_duration,
            )
            if not center
            else _core.asymmetric_levy_fpt_central_moment(
                self.start_position,
                self.alpha,
                self.beta,
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

    def occupation_time_moment(
        self,
        domain: tuple[real, real],
        duration: real,
        order: int,
        center: bool = False,
        particles: int = 10_000,
        step_size: float = 0.01,
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

        result = (
            _core.asymmetric_levy_occupation_time_raw_moment(
                self.start_position,
                self.alpha,
                self.beta,
                (_a, _b),
                _order,
                _particles,
                _step_size,
                _duration,
            )
            if not center
            else _core.asymmetric_levy_occupation_time_central_moment(
                self.start_position,
                self.alpha,
                self.beta,
                (_a, _b),
                _duration,
                _order,
                _particles,
                _step_size,
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
        step_size: float = 0.01,
        quad_order: int = 10,
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
