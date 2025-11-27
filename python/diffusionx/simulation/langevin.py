from typing import Callable
from diffusionx import _core
from .basic import real, Vector
from .utils import (
    ensure_float,
    validate_bool,
    validate_domain,
    validate_order,
    validate_particles,
    validate_positive_float,
    validate_positive_integer,
)


class Langevin:
    """
    Langevin equation:

    dx(t) = f(x(t), t) dt + g(x(t), t) dW(t), x(0) = x0

    The underlying implementation has been optimized with an efficient callback mechanism
    for interaction between Python functions and Rust code.

    Parameters
    ----------
    drift_func : Callable[[float, float], float]
        Drift function f(x, t).
    diffusion_func : Callable[[float, float], float]
        Diffusion function g(x, t).
    start_position : float, optional
        Initial position x0. Defaults to 0.0.
    """

    def __init__(
        self,
        drift_func: Callable[[float, float], float],
        diffusion_func: Callable[[float, float], float],
        start_position: real = 0.0,
    ) -> None:
        if not callable(drift_func):
            raise TypeError(
                f"drift_func must be a callable function, got {type(drift_func).__name__}"
            )
        if not callable(diffusion_func):
            raise TypeError(
                f"diffusion_func must be a callable function, got {type(diffusion_func).__name__}"
            )

        try:
            start_position = ensure_float(start_position)
        except TypeError as e:
            raise TypeError(f"start_position must be a number. Error: {e}") from e

        self.drift_func = drift_func
        self.diffusion_func = diffusion_func
        self.start_position = start_position

    def simulate(self, duration: real, time_step: real) -> tuple[Vector, Vector]:
        """
        Simulate the Langevin process.

        Parameters
        ----------
        duration : real
            Simulation duration (must be positive).
        time_step : real
            Time step size (must be positive).

        Raises
        ------
        TypeError
            If duration or time_step are not numbers.
        ValueError
            If duration or time_step are not positive.
        NotImplementedError
            If the corresponding _core function is not found.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (time points array, position array)
        """
        if not hasattr(_core, "langevin_simulate"):
            raise NotImplementedError(
                "langevin_simulate not implemented in _core module"
            )
        try:
            duration = ensure_float(duration)
            time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e

        if duration <= 0:
            raise ValueError("duration must be positive")
        if time_step <= 0:
            raise ValueError("time_step must be positive")

        return _core.langevin_simulate(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            duration,
            time_step,
        )

    def moment(
        self,
        duration: real,
        order: int,
        central: bool = True,
        particles: int = 10_000,
        time_step: real = 0.01,
    ) -> float:
        """
        Calculate the moment of the Langevin process.

        Parameters
        ----------
        duration : real
            Simulation duration (must be positive).
        order : int
            Order of the moment (must be non-negative).
        particles : int
            Number of particles to simulate (must be positive).
        time_step : real
            Time step size (must be positive).

        Raises
        ------
        TypeError
            If duration, order, particles, or time_step have incorrect types.
        ValueError
            If parameters have invalid values.
        NotImplementedError
            If the corresponding _core function is not found.

        Returns
        -------
        float
            Raw moment.
        """
        if not hasattr(_core, "langevin_raw_moment"):
            raise NotImplementedError(
                "langevin_raw_moment not implemented in _core module"
            )

        validate_bool(central, "central")
        order = validate_order(order)
        particles = validate_particles(particles)
        duration = validate_positive_float(duration, "duration")
        time_step = validate_positive_float(time_step, "time_step")

        if order == 0:
            return 1.0

        result = (
            _core.langevin_raw_moment(
                self.drift_func,
                self.diffusion_func,
                self.start_position,
                duration,
                order,
                particles,
                time_step,
            )
            if not central
            else _core.langevin_central_moment(
                self.drift_func,
                self.diffusion_func,
                self.start_position,
                duration,
                order,
                particles,
                time_step,
            )
        )

        return result

    def fpt(
        self,
        domain: tuple[real, real],
        time_step: float = 0.01,
        max_duration: real = 1000,
    ) -> float | None:
        a, b = validate_domain(domain, process_name="Langevin FPT")
        time_step = validate_positive_float(time_step, "time_step")
        max_duration = validate_positive_float(max_duration, "max_duration")

        return _core.langevin_fpt(
            self.start_position,
            self.drift_func,
            self.diffusion_func,
            time_step,
            (a, b),
            max_duration,
        )

    def fpt_moment(
        self,
        domain: tuple[real, real],
        order: int,
        central: bool = False,
        particles: int = 10_000,
        time_step: real = 0.01,
        max_duration: real = 1000,
    ) -> float | None:
        validate_bool(central, "central")
        a, b = validate_domain(domain, process_name="Langevin FPT raw moment")
        order = validate_order(order)
        particles = validate_particles(particles)
        time_step = validate_positive_float(time_step, "time_step")
        max_duration = validate_positive_float(max_duration, "max_duration")

        result = (
            _core.langevin_fpt_raw_moment(
                self.start_position,
                self.drift_func,
                self.diffusion_func,
                (a, b),
                order,
                particles,
                time_step,
                max_duration,
            )
            if not central
            else _core.langevin_fpt_central_moment(
                self.start_position,
                self.drift_func,
                self.diffusion_func,
                (a, b),
                order,
                particles,
                time_step,
                max_duration,
            )
        )
        return result

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
        time_step: real = 0.01,
    ) -> float:
        a, b = validate_domain(domain, process_name="Langevin Occupation Time")
        duration = validate_positive_float(duration, "duration")
        time_step = validate_positive_float(time_step, "time_step")

        return _core.langevin_occupation_time(
            self.start_position,
            self.drift_func,
            self.diffusion_func,
            time_step,
            (a, b),
            duration,
        )

    def occupation_time_moment(
        self,
        domain: tuple[real, real],
        duration: real,
        order: int,
        central: bool = False,
        particles: int = 10_000,
        time_step: real = 0.01,
    ) -> float:
        validate_bool(central, "central")
        a, b = validate_domain(domain, process_name="Langevin Occupation raw moment")
        order = validate_order(order)
        particles = validate_particles(particles)
        duration = validate_positive_float(duration, "duration")
        time_step = validate_positive_float(time_step, "time_step")

        if order == 0:
            return 1.0

        result = (
            _core.langevin_occupation_time_raw_moment(
                self.start_position,
                self.drift_func,
                self.diffusion_func,
                (a, b),
                order,
                particles,
                time_step,
                duration,
            )
            if not central
            else _core.langevin_occupation_time_central_moment(
                self.start_position,
                self.drift_func,
                self.diffusion_func,
                (a, b),
                order,
                particles,
                time_step,
                duration,
            )
        )
        return result

    def tamsd(
        self,
        duration: real,
        delta: real,
        time_step: float = 0.01,
        quad_order: int = 10,
    ) -> float:
        duration = validate_positive_float(duration, "duration")
        delta = validate_positive_float(delta, "delta")
        time_step = validate_positive_float(time_step, "time_step")
        quad_order = validate_positive_integer(quad_order, "quad_order")

        return _core.langevin_tamsd(
            self.drift_func,
            self.diffusion_func,
            duration,
            delta,
            time_step,
            quad_order,
        )

    def eatamsd(
        self,
        duration: real,
        delta: real,
        particles: int = 10_000,
        time_step: float = 0.01,
        quad_order: int = 10,
    ) -> float:
        duration = validate_positive_float(duration, "duration")
        delta = validate_positive_float(delta, "delta")
        particles = validate_particles(particles)
        time_step = validate_positive_float(time_step, "time_step")
        quad_order = validate_positive_integer(quad_order, "quad_order")

        return _core.langevin_eatamsd(
            self.drift_func,
            self.diffusion_func,
            duration,
            delta,
            particles,
            time_step,
            quad_order,
        )


class GeneralizedLangevin:
    """
    Generalized Langevin equation:

    dx(t) = f(x(t), t) dt + g(x(t), t) dL_alpha(t), x(0) = x0

    dL_alpha(t) is a Lévy stable noise source.

    Parameters
    ----------
    drift_func : Callable[[float, float], float]
        Drift function f(x, t).
    diffusion_func : Callable[[float, float], float]
        Diffusion function g(x, t).
    start_position : real, optional
        Initial position x0. Defaults to 0.0.
    alpha : real, optional
        Stability index of the stable noise (0, 2]. Defaults to 1.5.
    """

    def __init__(
        self,
        drift_func: Callable[[float, float], float],
        diffusion_func: Callable[[float, float], float],
        start_position: real = 0.0,
        alpha: real = 1.5,
    ) -> None:
        if not callable(drift_func):
            raise TypeError(
                f"drift_func must be a callable function, got {type(drift_func).__name__}"
            )
        if not callable(diffusion_func):
            raise TypeError(
                f"diffusion_func must be a callable function, got {type(diffusion_func).__name__}"
            )

        try:
            start_position = ensure_float(start_position)
            alpha = ensure_float(alpha)
        except TypeError as e:
            raise TypeError(
                f"start_position and alpha must be numbers. Error: {e}"
            ) from e

        if not (0 < alpha <= 2):
            raise ValueError(
                f"alpha (stability index) must be in the range (0, 2], got {alpha}"
            )

        self.drift_func = drift_func
        self.diffusion_func = diffusion_func
        self.start_position = start_position
        self.alpha = alpha

    def simulate(
        self, duration: real, time_step: float = 0.01
    ) -> tuple[Vector, Vector]:
        """
        Simulate the Generalized Langevin process.

        Parameters
        ----------
        duration : real
            Simulation duration (must be positive).
        time_step : real
            Time step size (must be positive).

        Raises
        ------
        TypeError
            If duration or time_step are not numbers.
        ValueError
            If duration or time_step are not positive.
        NotImplementedError
            If the corresponding _core function is not found.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (time points array, position array)
        """
        if not hasattr(_core, "generalized_langevin_simulate"):
            raise NotImplementedError(
                "generalized_langevin_simulate not implemented in _core module"
            )
        try:
            duration = ensure_float(duration)
            time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e

        if duration <= 0:
            raise ValueError("duration must be positive")
        if time_step <= 0:
            raise ValueError("time_step must be positive")

        return _core.generalized_langevin_simulate(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            self.alpha,
            duration,
            time_step,
        )

    def moment(
        self,
        duration: real,
        order: int,
        central: bool = True,
        particles: int = 10_000,
        time_step: real = 0.01,
    ) -> float:
        """
        Calculate the moment of the Generalized Langevin process.

        Parameters
        ----------
        duration : real
            Simulation duration (must be positive).
        order : int
            Order of the moment (must be non-negative).
        particles : int
            Number of particles (must be positive).
        time_step : real
            Time step size (must be positive).

        Raises
        ------
        TypeError
            If parameters have incorrect types.
        ValueError
            If parameters have invalid values.
        NotImplementedError
            If the corresponding _core function is not found.

        Returns
        -------
        float
            Raw moment.
        """
        if not hasattr(_core, "generalized_langevin_raw_moment"):
            raise NotImplementedError(
                "generalized_langevin_raw_moment not implemented in _core module"
            )
        validate_bool(central, "central")
        order = validate_order(order)
        particles = validate_particles(particles)
        duration = validate_positive_float(duration, "duration")
        time_step = validate_positive_float(time_step, "time_step")

        if order == 0:
            return 1.0

        result = (
            _core.generalized_langevin_raw_moment(
                self.drift_func,
                self.diffusion_func,
                self.start_position,
                self.alpha,
                duration,
                order,
                particles,
                time_step,
            )
            if not central
            else _core.generalized_langevin_central_moment(
                self.drift_func,
                self.diffusion_func,
                self.start_position,
                self.alpha,
                duration,
                order,
                particles,
                time_step,
            )
        )
        return result

    def fpt(
        self,
        domain: tuple[real, real],
        time_step: float = 0.01,
        max_duration: real = 1000,
    ) -> float | None:
        a, b = validate_domain(domain, process_name="GeneralizedLangevin FPT")
        time_step = validate_positive_float(time_step, "time_step")
        max_duration = validate_positive_float(max_duration, "max_duration")

        return _core.generalized_langevin_fpt(
            self.start_position,
            self.drift_func,
            self.diffusion_func,
            self.alpha,
            time_step,
            (a, b),
            max_duration,
        )

    def fpt_moment(
        self,
        domain: tuple[real, real],
        order: int,
        central: bool = False,
        particles: int = 10_000,
        time_step: float = 0.01,
        max_duration: real = 1000,
    ) -> float | None:
        validate_bool(central, "central")
        a, b = validate_domain(
            domain, process_name="GeneralizedLangevin FPT raw moment"
        )
        order = validate_order(order)
        particles = validate_particles(particles)
        time_step = validate_positive_float(time_step, "time_step")
        max_duration = validate_positive_float(max_duration, "max_duration")

        result = (
            _core.generalized_langevin_fpt_raw_moment(
                self.start_position,
                self.drift_func,
                self.diffusion_func,
                self.alpha,
                (a, b),
                order,
                particles,
                time_step,
                max_duration,
            )
            if not central
            else _core.generalized_langevin_fpt_central_moment(
                self.start_position,
                self.drift_func,
                self.diffusion_func,
                self.alpha,
                (a, b),
                order,
                particles,
                time_step,
                max_duration,
            )
        )
        return result

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
        time_step: real = 0.01,
    ) -> float:
        a, b = validate_domain(
            domain, process_name="GeneralizedLangevin Occupation Time"
        )
        duration = validate_positive_float(duration, "duration")
        time_step = validate_positive_float(time_step, "time_step")

        return _core.generalized_langevin_occupation_time(
            self.start_position,
            self.drift_func,
            self.diffusion_func,
            self.alpha,
            time_step,
            (a, b),
            duration,
        )

    def occupation_time_moment(
        self,
        domain: tuple[real, real],
        duration: real,
        order: int,
        central: bool = False,
        particles: int = 10_000,
        time_step: real = 0.01,
    ) -> float:
        validate_bool(central, "central")
        a, b = validate_domain(
            domain, process_name="GeneralizedLangevin Occupation raw moment"
        )
        order = validate_order(order)
        particles = validate_particles(particles)
        duration = validate_positive_float(duration, "duration")
        time_step = validate_positive_float(time_step, "time_step")

        if order == 0:
            return 1.0

        result = (
            _core.generalized_langevin_occupation_time_raw_moment(
                self.start_position,
                self.drift_func,
                self.diffusion_func,
                self.alpha,
                (a, b),
                order,
                particles,
                time_step,
                duration,
            )
            if not central
            else _core.generalized_langevin_occupation_time_central_moment(
                self.start_position,
                self.drift_func,
                self.diffusion_func,
                self.alpha,
                (a, b),
                order,
                particles,
                time_step,
                duration,
            )
        )
        return result

    def tamsd(
        self,
        duration: real,
        delta: real,
        time_step: float = 0.01,
        quad_order: int = 10,
    ) -> float:
        duration = validate_positive_float(duration, "duration")
        delta = validate_positive_float(delta, "delta")
        time_step = validate_positive_float(time_step, "time_step")
        quad_order = validate_positive_integer(quad_order, "quad_order")

        return _core.generalized_langevin_tamsd(
            self.drift_func,
            self.diffusion_func,
            self.alpha,
            duration,
            delta,
            time_step,
            quad_order,
        )

    def eatamsd(
        self,
        duration: real,
        delta: real,
        particles: int = 10_000,
        time_step: float = 0.01,
        quad_order: int = 10,
    ) -> float:
        duration = validate_positive_float(duration, "duration")
        delta = validate_positive_float(delta, "delta")
        particles = validate_particles(particles)
        time_step = validate_positive_float(time_step, "time_step")
        quad_order = validate_positive_integer(quad_order, "quad_order")

        return _core.generalized_langevin_eatamsd(
            self.drift_func,
            self.diffusion_func,
            self.alpha,
            duration,
            delta,
            particles,
            time_step,
            quad_order,
        )


class SubordinatedLangevin:
    """
    Subordinated Langevin equation driven by a subordinator process for operational time.

    Parameters
    ----------
    drift_func : Callable[[float, float], float]
        Drift function f(x, u) where u is operational time.
    diffusion_func : Callable[[float, float], float]
        Diffusion function g(x, u) where u is operational time.
    subordinator_alpha : real, optional
        Alpha parameter for the subordinator (0, 1). Defaults to 0.7.
    start_position : real, optional
        Initial position x0. Defaults to 0.0.
    """

    def __init__(
        self,
        drift_func: Callable[[float, float], float],
        diffusion_func: Callable[[float, float], float],
        subordinator_alpha: real = 0.7,
        start_position: real = 0.0,
    ) -> None:
        if not callable(drift_func):
            raise TypeError(
                f"drift_func must be a callable function, got {type(drift_func).__name__}"
            )
        if not callable(diffusion_func):
            raise TypeError(
                f"diffusion_func must be a callable function, got {type(diffusion_func).__name__}"
            )

        try:
            start_position = ensure_float(start_position)
            subordinator_alpha = ensure_float(subordinator_alpha)
        except TypeError as e:
            raise TypeError(
                f"start_position and subordinator_alpha must be numbers. Error: {e}"
            ) from e

        if not (0 < subordinator_alpha < 1):
            raise ValueError(
                f"subordinator_alpha must be in the range (0, 1), got {subordinator_alpha}"
            )

        self.drift_func = drift_func
        self.diffusion_func = diffusion_func
        self.subordinator_alpha = subordinator_alpha
        self.start_position = start_position

    def simulate(
        self, duration: real, time_step: float = 0.01
    ) -> tuple[Vector, Vector]:
        """
        Simulate the Subordinated Langevin process.
        (Parameters, Raises, Returns are similar to Langevin.simulate)
        """
        if not hasattr(_core, "subordinated_langevin_simulate"):
            raise NotImplementedError(
                "subordinated_langevin_simulate not implemented in _core module"
            )
        try:
            duration = ensure_float(duration)
            time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e

        if duration <= 0:
            raise ValueError("duration must be positive")
        if time_step <= 0:
            raise ValueError("time_step must be positive")

        return _core.subordinated_langevin_simulate(
            self.drift_func,
            self.diffusion_func,
            self.subordinator_alpha,
            self.start_position,
            duration,
            time_step,
        )

    def moment(
        self,
        duration: real,
        order: int,
        central: bool = True,
        particles: int = 10_000,
        time_step: float = 0.01,
    ) -> float:
        """
        Calculate the raw moment of the Subordinated Langevin process.
        (Parameters, Raises, Returns are similar to Langevin.raw_moment)
        """
        if not hasattr(_core, "subordinated_langevin_raw_moment"):
            raise NotImplementedError(
                "subordinated_langevin_raw_moment not implemented in _core module"
            )
        validate_bool(central, "central")
        order = validate_order(order)
        particles = validate_particles(particles)
        duration = validate_positive_float(duration, "duration")
        time_step = validate_positive_float(time_step, "time_step")

        if order == 0:
            return 1.0

        result = (
            _core.subordinated_langevin_raw_moment(
                self.drift_func,
                self.diffusion_func,
                self.subordinator_alpha,
                self.start_position,
                duration,
                order,
                particles,
                time_step,
            )
            if not central
            else _core.subordinated_langevin_central_moment(
                self.drift_func,
                self.diffusion_func,
                self.subordinator_alpha,
                self.start_position,
                duration,
            )
        )
        return result

    def fpt(
        self, domain: tuple[real, real], max_duration: real = 1000
    ) -> float | None:  # No time_step here as per _core
        a, b = validate_domain(domain, process_name="SubordinatedLangevin FPT")
        max_duration = validate_positive_float(max_duration, "max_duration")

        return _core.subordinated_langevin_fpt(
            self.start_position,
            self.drift_func,
            self.diffusion_func,
            self.subordinator_alpha,
            (a, b),
            max_duration,
        )

    def fpt_moment(
        self,
        domain: tuple[real, real],
        order: int,
        central: bool = True,
        particles: int = 10_000,
        time_step: real = 0.01,
        max_duration: real = 1000,
    ) -> float | None:
        validate_bool(central, "central")
        a, b = validate_domain(
            domain, process_name="SubordinatedLangevin FPT raw moment"
        )
        order = validate_order(order)
        particles = validate_particles(particles)
        time_step = validate_positive_float(time_step, "time_step")
        max_duration = validate_positive_float(max_duration, "max_duration")

        result = (
            _core.subordinated_langevin_fpt_raw_moment(
                self.start_position,
                self.drift_func,
                self.diffusion_func,
                self.subordinator_alpha,
                (a, b),
                order,
                particles,
                time_step,
                max_duration,
            )
            if not central
            else _core.subordinated_langevin_fpt_central_moment(
                self.start_position,
                self.drift_func,
                self.diffusion_func,
                self.subordinator_alpha,
                (a, b),
                order,
                particles,
                time_step,
                max_duration,
            )
        )
        return result

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
        time_step: float = 0.01,
    ) -> float:
        a, b = validate_domain(
            domain, process_name="SubordinatedLangevin Occupation Time"
        )
        duration = validate_positive_float(duration, "duration")
        time_step = validate_positive_float(time_step, "time_step")

        return _core.subordinated_langevin_occupation_time(
            self.start_position,
            self.drift_func,
            self.diffusion_func,
            self.subordinator_alpha,
            time_step,
            (a, b),
            duration,
        )

    def occupation_time_moment(
        self,
        domain: tuple[real, real],
        duration: real,
        order: int,
        central: bool = True,
        particles: int = 10_000,
        time_step: float = 0.01,
    ) -> float:
        validate_bool(central, "central")
        a, b = validate_domain(
            domain, process_name="SubordinatedLangevin Occupation raw moment"
        )
        order = validate_order(order)
        particles = validate_particles(particles)
        duration = validate_positive_float(duration, "duration")
        time_step = validate_positive_float(time_step, "time_step")

        if order == 0:
            return 1.0

        result = (
            _core.subordinated_langevin_occupation_time_raw_moment(
                self.start_position,
                self.drift_func,
                self.diffusion_func,
                self.subordinator_alpha,
                (a, b),
                order,
                particles,
                time_step,
                duration,
            )
            if not central
            else _core.subordinated_langevin_occupation_time_central_moment(
                self.start_position,
                self.drift_func,
                self.diffusion_func,
                self.subordinator_alpha,
                (a, b),
                order,
                particles,
                time_step,
                duration,
            )
        )
        return result

    def tamsd(
        self,
        duration: real,
        delta: real,
        time_step: float = 0.01,
        quad_order: int = 10,
    ) -> float:
        _duration = validate_positive_float(duration, "duration")
        _delta = validate_positive_float(delta, "delta")
        _time_step = validate_positive_float(time_step, "time_step")

        return _core.subordinated_langevin_tamsd(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            self.subordinator_alpha,
            _duration,
            _delta,
            _time_step,
            quad_order,
        )

    def eatamsd(
        self,
        duration: real,
        delta: real,
        particles: int = 10_000,
        time_step: float = 0.01,
        quad_order: int = 10,
    ) -> float:
        _duration = validate_positive_float(duration, "duration")
        _delta = validate_positive_float(delta, "delta")
        _particles = validate_particles(particles)
        _time_step = validate_positive_float(time_step, "time_step")

        return _core.subordinated_langevin_eatamsd(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            self.subordinator_alpha,
            _duration,
            _delta,
            _particles,
            _time_step,
            quad_order,
        )
