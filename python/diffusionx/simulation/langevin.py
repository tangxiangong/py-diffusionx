from typing import Callable, Union
from numpy import ndarray
from .. import _core
from ..simulation.utils import ensure_float

# Define real for consistency if used, though parameters are often explicitly float/int here
real = Union[float, int]


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
            _start_position = ensure_float(start_position)
        except TypeError as e:
            raise TypeError(f"start_position must be a number. Error: {e}") from e

        self.drift_func = drift_func
        self.diffusion_func = diffusion_func
        self.start_position = _start_position

    def simulate(self, duration: real, time_step: real) -> tuple[ndarray, ndarray]:
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
            _duration = ensure_float(duration)
            _time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e

        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _time_step <= 0:
            raise ValueError("time_step must be positive")

        return _core.langevin_simulate(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            _duration,
            _time_step,
        )

    def raw_moment(
        self, duration: real, order: int, particles: int, time_step: real
    ) -> float:
        """
        Calculate the raw moment of the Langevin process.

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

        if not isinstance(order, int):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError("order must be non-negative")

        if not isinstance(particles, int):
            raise TypeError(
                f"particles must be an integer, got {type(particles).__name__}"
            )
        if particles <= 0:
            raise ValueError("particles must be positive")

        try:
            _duration = ensure_float(duration)
            _time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e

        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _time_step <= 0:
            raise ValueError("time_step must be positive")

        if order == 0:
            return 1.0

        return _core.langevin_raw_moment(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            _duration,
            order,
            particles,
            _time_step,
        )

    def central_moment(
        self, duration: real, order: int, particles: int, time_step: real
    ) -> float:
        """
        Calculate the central moment of the Langevin process.

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
            Central moment.
        """
        if not hasattr(_core, "langevin_central_moment"):
            raise NotImplementedError(
                "langevin_central_moment not implemented in _core module"
            )

        if not isinstance(order, int):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError("order must be non-negative")

        if not isinstance(particles, int):
            raise TypeError(
                f"particles must be an integer, got {type(particles).__name__}"
            )
        if particles <= 0:
            raise ValueError("particles must be positive")

        try:
            _duration = ensure_float(duration)
            _time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e

        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _time_step <= 0:
            raise ValueError("time_step must be positive")

        if order == 0:
            return 1.0
        if order == 1:
            # The first central moment is E[X - E[X]], which is 0 by definition.
            return 0.0

        return _core.langevin_central_moment(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            _duration,
            order,
            particles,
            _time_step,
        )

    def mean(self, duration: real, particles: int, time_step: real) -> float:
        """
        Calculate the mean of the Langevin process.

        Parameters
        ----------
        duration : real
            Simulation duration (must be positive).
        particles : int
            Number of particles to simulate (must be positive).
        time_step : real
            Time step size (must be positive).

        Raises
        ------
        TypeError
            If duration, particles, or time_step have incorrect types.
        ValueError
            If parameters have invalid values.

        Returns
        -------
        float
            Mean value.
        """
        # Validation for order=1 is implicitly handled by raw_moment structure
        # We still need to validate duration, particles, time_step here before passing.
        if not isinstance(particles, int):
            raise TypeError(
                f"particles must be an integer, got {type(particles).__name__}"
            )
        if particles <= 0:
            raise ValueError("particles must be positive")
        try:
            _duration = ensure_float(duration)
            _time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e
        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _time_step <= 0:
            raise ValueError("time_step must be positive")

        return self.raw_moment(_duration, 1, particles, _time_step)

    def variance(self, duration: real, particles: int, time_step: real) -> float:
        """
        Calculate the variance of the Langevin process.

        Parameters
        ----------
        duration : real
            Simulation duration (must be positive).
        particles : int
            Number of particles to simulate (must be positive).
        time_step : real
            Time step size (must be positive).

        Raises
        ------
        TypeError
            If duration, particles, or time_step have incorrect types.
        ValueError
            If parameters have invalid values.

        Returns
        -------
        float
            Variance.
        """
        # Validation for order=2 is implicitly handled by central_moment structure
        # We still need to validate duration, particles, time_step here before passing.
        if not isinstance(particles, int):
            raise TypeError(
                f"particles must be an integer, got {type(particles).__name__}"
            )
        if particles <= 0:
            raise ValueError("particles must be positive")
        try:
            _duration = ensure_float(duration)
            _time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e
        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _time_step <= 0:
            raise ValueError("time_step must be positive")

        return self.central_moment(_duration, 2, particles, _time_step)


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
            _start_position = ensure_float(start_position)
            _alpha = ensure_float(alpha)
        except TypeError as e:
            raise TypeError(
                f"start_position and alpha must be numbers. Error: {e}"
            ) from e

        if not (0 < _alpha <= 2):
            raise ValueError(
                f"alpha (stability index) must be in the range (0, 2], got {_alpha}"
            )

        self.drift_func = drift_func
        self.diffusion_func = diffusion_func
        self.start_position = _start_position
        self.alpha = _alpha

    def simulate(self, duration: real, time_step: real) -> tuple[ndarray, ndarray]:
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
            _duration = ensure_float(duration)
            _time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e

        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _time_step <= 0:
            raise ValueError("time_step must be positive")

        return _core.generalized_langevin_simulate(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            self.alpha,
            _duration,
            _time_step,
        )

    def raw_moment(
        self, duration: real, order: int, particles: int, time_step: real
    ) -> float:
        """
        Calculate the raw moment of the Generalized Langevin process.

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
        if not isinstance(order, int):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError("order must be non-negative")

        if not isinstance(particles, int):
            raise TypeError(
                f"particles must be an integer, got {type(particles).__name__}"
            )
        if particles <= 0:
            raise ValueError("particles must be positive")

        try:
            _duration = ensure_float(duration)
            _time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e

        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _time_step <= 0:
            raise ValueError("time_step must be positive")

        if order == 0:
            return 1.0

        return _core.generalized_langevin_raw_moment(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            self.alpha,
            _duration,
            order,
            particles,
            _time_step,
        )

    def central_moment(
        self, duration: real, order: int, particles: int, time_step: real
    ) -> float:
        """
        Calculate the central moment of the Generalized Langevin process.

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
            Central moment.
        """
        if not hasattr(_core, "generalized_langevin_central_moment"):
            raise NotImplementedError(
                "generalized_langevin_central_moment not implemented in _core module"
            )
        if not isinstance(order, int):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError("order must be non-negative")

        if not isinstance(particles, int):
            raise TypeError(
                f"particles must be an integer, got {type(particles).__name__}"
            )
        if particles <= 0:
            raise ValueError("particles must be positive")

        try:
            _duration = ensure_float(duration)
            _time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e

        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _time_step <= 0:
            raise ValueError("time_step must be positive")

        if order == 0:
            return 1.0
        if order == 1:
            return 0.0  # First central moment is 0

        return _core.generalized_langevin_central_moment(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            self.alpha,
            _duration,
            order,
            particles,
            _time_step,
        )

    def mean(self, duration: real, particles: int, time_step: real) -> float:
        """
        Calculate the mean of the Generalized Langevin process.
        (Parameters, Raises, Returns are similar to Langevin.mean)
        """
        if not isinstance(particles, int):
            raise TypeError(
                f"particles must be an integer, got {type(particles).__name__}"
            )
        if particles <= 0:
            raise ValueError("particles must be positive")
        try:
            _duration = ensure_float(duration)
            _time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e
        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _time_step <= 0:
            raise ValueError("time_step must be positive")
        return self.raw_moment(_duration, 1, particles, _time_step)

    def variance(self, duration: real, particles: int, time_step: real) -> float:
        """
        Calculate the variance of the Generalized Langevin process.
        (Parameters, Raises, Returns are similar to Langevin.variance)
        """
        if not isinstance(particles, int):
            raise TypeError(
                f"particles must be an integer, got {type(particles).__name__}"
            )
        if particles <= 0:
            raise ValueError("particles must be positive")
        try:
            _duration = ensure_float(duration)
            _time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e
        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _time_step <= 0:
            raise ValueError("time_step must be positive")
        return self.central_moment(_duration, 2, particles, _time_step)


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
            _start_position = ensure_float(start_position)
            _subordinator_alpha = ensure_float(subordinator_alpha)
        except TypeError as e:
            raise TypeError(
                f"start_position and subordinator_alpha must be numbers. Error: {e}"
            ) from e

        if not (0 < _subordinator_alpha < 1):
            raise ValueError(
                f"subordinator_alpha must be in the range (0, 1), got {_subordinator_alpha}"
            )

        self.drift_func = drift_func
        self.diffusion_func = diffusion_func
        self.subordinator_alpha = _subordinator_alpha
        self.start_position = _start_position

    def simulate(self, duration: real, time_step: real) -> tuple[ndarray, ndarray]:
        """
        Simulate the Subordinated Langevin process.
        (Parameters, Raises, Returns are similar to Langevin.simulate)
        """
        if not hasattr(_core, "subordinated_langevin_simulate"):
            raise NotImplementedError(
                "subordinated_langevin_simulate not implemented in _core module"
            )
        try:
            _duration = ensure_float(duration)
            _time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e

        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _time_step <= 0:
            raise ValueError("time_step must be positive")

        return _core.subordinated_langevin_simulate(
            self.drift_func,
            self.diffusion_func,
            self.subordinator_alpha,
            self.start_position,
            _duration,
            _time_step,
        )

    def raw_moment(
        self, duration: real, order: int, particles: int, time_step: real
    ) -> float:
        """
        Calculate the raw moment of the Subordinated Langevin process.
        (Parameters, Raises, Returns are similar to Langevin.raw_moment)
        """
        if not hasattr(_core, "subordinated_langevin_raw_moment"):
            raise NotImplementedError(
                "subordinated_langevin_raw_moment not implemented in _core module"
            )
        if not isinstance(order, int):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError("order must be non-negative")

        if not isinstance(particles, int):
            raise TypeError(
                f"particles must be an integer, got {type(particles).__name__}"
            )
        if particles <= 0:
            raise ValueError("particles must be positive")

        try:
            _duration = ensure_float(duration)
            _time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e

        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _time_step <= 0:
            raise ValueError("time_step must be positive")

        if order == 0:
            return 1.0

        return _core.subordinated_langevin_raw_moment(
            self.drift_func,
            self.diffusion_func,
            self.subordinator_alpha,
            self.start_position,
            _duration,
            order,
            particles,
            _time_step,
        )

    def central_moment(
        self, duration: real, order: int, particles: int, time_step: real
    ) -> float:
        """
        Calculate the central moment of the Subordinated Langevin process.
        (Parameters, Raises, Returns are similar to Langevin.central_moment)
        """
        if not hasattr(_core, "subordinated_langevin_central_moment"):
            raise NotImplementedError(
                "subordinated_langevin_central_moment not implemented in _core module"
            )
        if not isinstance(order, int):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError("order must be non-negative")

        if not isinstance(particles, int):
            raise TypeError(
                f"particles must be an integer, got {type(particles).__name__}"
            )
        if particles <= 0:
            raise ValueError("particles must be positive")

        try:
            _duration = ensure_float(duration)
            _time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e

        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _time_step <= 0:
            raise ValueError("time_step must be positive")

        if order == 0:
            return 1.0
        if order == 1:
            return 0.0  # First central moment is 0

        return _core.subordinated_langevin_central_moment(
            self.drift_func,
            self.diffusion_func,
            self.subordinator_alpha,
            self.start_position,
            _duration,
            order,
            particles,
            _time_step,
        )

    def mean(self, duration: real, particles: int, time_step: real) -> float:
        """
        Calculate the mean of the Subordinated Langevin process.
        (Parameters, Raises, Returns are similar to Langevin.mean)
        """
        if not isinstance(particles, int):
            raise TypeError(
                f"particles must be an integer, got {type(particles).__name__}"
            )
        if particles <= 0:
            raise ValueError("particles must be positive")
        try:
            _duration = ensure_float(duration)
            _time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e
        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _time_step <= 0:
            raise ValueError("time_step must be positive")
        return self.raw_moment(_duration, 1, particles, _time_step)

    def variance(self, duration: real, particles: int, time_step: real) -> float:
        """
        Calculate the variance of the Subordinated Langevin process.
        (Parameters, Raises, Returns are similar to Langevin.variance)
        """
        if not isinstance(particles, int):
            raise TypeError(
                f"particles must be an integer, got {type(particles).__name__}"
            )
        if particles <= 0:
            raise ValueError("particles must be positive")
        try:
            _duration = ensure_float(duration)
            _time_step = ensure_float(time_step)
        except TypeError as e:
            raise TypeError(
                f"duration and time_step must be numbers. Error: {e}"
            ) from e
        if _duration <= 0:
            raise ValueError("duration must be positive")
        if _time_step <= 0:
            raise ValueError("time_step must be positive")
        return self.central_moment(_duration, 2, particles, _time_step)
