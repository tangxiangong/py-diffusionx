from typing import Callable, Tuple
from numpy import ndarray
from .. import _core


class Langevin:
    """
    Langevin equation:

    dx(t) = f(x(t), t) dt + g(x(t), t) dW(t), x(0) = x0

    The underlying implementation has been optimized with an efficient callback mechanism
    for interaction between Python functions and Rust code.

    Parameters
    ----------
    drift_func : Callable[[float, float], float]
        Drift function f(x, t)
    diffusion_func : Callable[[float, float], float]
        Diffusion function g(x, t)
    start_position : float
        Initial position x0
    """

    def __init__(
        self,
        drift_func: Callable[[float, float], float],
        diffusion_func: Callable[[float, float], float],
        start_position: float = 0.0,
    ) -> None:
        self.drift_func = drift_func
        self.diffusion_func = diffusion_func
        self.start_position = start_position

    def simulate(self, duration: float, time_step: float) -> Tuple[ndarray, ndarray]:
        """
        Simulate the Langevin process

        Parameters
        ----------
        duration : float
            Simulation duration
        time_step : float
            Time step size

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray]
            (time points array, position array)
        """
        if not hasattr(_core, "langevin_simulate"):
            raise NotImplementedError(
                "langevin_simulate not implemented in _core module"
            )

        return _core.langevin_simulate(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            duration,
            time_step,
        )

    def raw_moment(
        self, duration: float, order: int, particles: int, time_step: float
    ) -> float:
        """
        Calculate the raw moment of the Langevin process

        Parameters
        ----------
        duration : float
            Simulation duration
        order : int
            Order of the moment
        particles : int
            Number of particles to simulate
        time_step : float
            Time step size

        Returns
        -------
        float
            Raw moment
        """
        if not hasattr(_core, "langevin_raw_moment"):
            raise NotImplementedError(
                "langevin_raw_moment not implemented in _core module"
            )

        return _core.langevin_raw_moment(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            duration,
            order,
            particles,
            time_step,
        )

    def central_moment(
        self, duration: float, order: int, particles: int, time_step: float
    ) -> float:
        """
        Calculate the central moment of the Langevin process

        Parameters
        ----------
        duration : float
            Simulation duration
        order : int
            Order of the moment
        particles : int
            Number of particles to simulate
        time_step : float
            Time step size

        Returns
        -------
        float
            Central moment
        """
        if not hasattr(_core, "langevin_central_moment"):
            raise NotImplementedError(
                "langevin_central_moment not implemented in _core module"
            )

        return _core.langevin_central_moment(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            duration,
            order,
            particles,
            time_step,
        )

    def mean(self, duration: float, particles: int, time_step: float) -> float:
        """
        Calculate the mean of the Langevin process

        Parameters
        ----------
        duration : float
            Simulation duration
        particles : int
            Number of particles to simulate
        time_step : float
            Time step size

        Returns
        -------
        float
            Mean value
        """
        return self.raw_moment(duration, 1, particles, time_step)

    def variance(self, duration: float, particles: int, time_step: float) -> float:
        """
        Calculate the variance of the Langevin process

        Parameters
        ----------
        duration : float
            Simulation duration
        particles : int
            Number of particles to simulate
        time_step : float
            Time step size

        Returns
        -------
        float
            Variance
        """
        return self.central_moment(duration, 2, particles, time_step)


class GeneralizedLangevin:
    """
    Generalized Langevin equation:

    dx(t) = f(x(t), t) dt + g(x(t), t) dL_alpha(t), x(0) = x0

    The underlying implementation has been optimized with an efficient callback mechanism
    for interaction between Python functions and Rust code.

    Parameters
    ----------
    drift_func : Callable[[float, float], float]
        Drift function f(x, t)
    diffusion_func : Callable[[float, float], float]
        Diffusion function g(x, t)
    start_position : float
        Initial position x0
    alpha : float
        Stability index of the stable distribution
    """

    def __init__(
        self,
        drift_func: Callable[[float, float], float],
        diffusion_func: Callable[[float, float], float],
        start_position: float = 0.0,
        alpha: float = 1.5,
    ) -> None:
        self.drift_func = drift_func
        self.diffusion_func = diffusion_func
        self.start_position = start_position
        self.alpha = alpha

    def simulate(self, duration: float, time_step: float) -> Tuple[ndarray, ndarray]:
        """
        Simulate the Generalized Langevin process

        Parameters
        ----------
        duration : float
            Simulation duration
        time_step : float
            Time step size

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray]
            (time points array, position array)
        """
        if not hasattr(_core, "generalized_langevin_simulate"):
            raise NotImplementedError(
                "generalized_langevin_simulate not implemented in _core module"
            )

        return _core.generalized_langevin_simulate(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            self.alpha,
            duration,
            time_step,
        )

    def raw_moment(
        self, duration: float, order: int, particles: int, time_step: float
    ) -> float:
        """
        Calculate the raw moment of the Generalized Langevin process

        Parameters
        ----------
        duration : float
            Simulation duration
        order : int
            Order of the moment
        particles : int
            Number of particles to simulate
        time_step : float
            Time step size

        Returns
        -------
        float
            Raw moment
        """
        if not hasattr(_core, "generalized_langevin_raw_moment"):
            raise NotImplementedError(
                "generalized_langevin_raw_moment not implemented in _core module"
            )

        return _core.generalized_langevin_raw_moment(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            self.alpha,
            duration,
            order,
            particles,
            time_step,
        )

    def central_moment(
        self, duration: float, order: int, particles: int, time_step: float
    ) -> float:
        """
        Calculate the central moment of the Generalized Langevin process

        Parameters
        ----------
        duration : float
            Simulation duration
        order : int
            Order of the moment
        particles : int
            Number of particles to simulate
        time_step : float
            Time step size

        Returns
        -------
        float
            Central moment
        """
        if not hasattr(_core, "generalized_langevin_central_moment"):
            raise NotImplementedError(
                "generalized_langevin_central_moment not implemented in _core module"
            )

        return _core.generalized_langevin_central_moment(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            self.alpha,
            duration,
            order,
            particles,
            time_step,
        )

    def mean(self, duration: float, particles: int, time_step: float) -> float:
        """
        Calculate the mean of the Generalized Langevin process

        Parameters
        ----------
        duration : float
            Simulation duration
        particles : int
            Number of particles to simulate
        time_step : float
            Time step size

        Returns
        -------
        float
            Mean value
        """
        return self.raw_moment(duration, 1, particles, time_step)

    def variance(self, duration: float, particles: int, time_step: float) -> float:
        """
        Calculate the variance of the Generalized Langevin process

        Parameters
        ----------
        duration : float
            Simulation duration
        particles : int
            Number of particles to simulate
        time_step : float
            Time step size

        Returns
        -------
        float
            Variance
        """
        return self.central_moment(duration, 2, particles, time_step)


class SubordinatedLangevin:
    """
    Subordinated Langevin equation:

    dx(E(t)) = f(x(E(t)), E(t)) dE(t) + g(x(E(t)), E(t)) dW(E(t)), x(0) = x0

    where E(t) is the inverse subordinator process with stability index subordinator_alpha.

    The underlying implementation has been optimized with an efficient callback mechanism
    for interaction between Python functions and Rust code.

    Parameters
    ----------
    drift_func : Callable[[float, float], float]
        Drift function f(x, t)
    diffusion_func : Callable[[float, float], float]
        Diffusion function g(x, t)
    subordinator_alpha : float
        Stability index of the subordinator process (0 < alpha < 1)
    start_position : float
        Initial position x0
    """

    def __init__(
        self,
        drift_func: Callable[[float, float], float],
        diffusion_func: Callable[[float, float], float],
        subordinator_alpha: float = 0.7,
        start_position: float = 0.0,
    ) -> None:
        self.drift_func = drift_func
        self.diffusion_func = diffusion_func
        # Stored but not currently used in backend calls
        self.subordinator_alpha = subordinator_alpha
        self.start_position = start_position

    def simulate(self, duration: float, time_step: float) -> Tuple[ndarray, ndarray]:
        """
        Simulate the Subordinated Langevin process

        Parameters
        ----------
        duration : float
            Simulation duration
        time_step : float
            Time step size

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray]
            (time points array, position array)
        """
        if not hasattr(_core, "subordinated_langevin_simulate"):
            raise NotImplementedError(
                "subordinated_langevin_simulate not implemented in _core module"
            )

        return _core.subordinated_langevin_simulate(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            duration,
            time_step,
        )

    def raw_moment(
        self, duration: float, order: int, particles: int, time_step: float
    ) -> float:
        """
        Calculate the raw moment of the Subordinated Langevin process

        Parameters
        ----------
        duration : float
            Simulation duration
        order : int
            Order of the moment
        particles : int
            Number of particles to simulate
        time_step : float
            Time step size

        Returns
        -------
        float
            Raw moment
        """
        if not hasattr(_core, "subordinated_langevin_raw_moment"):
            raise NotImplementedError(
                "subordinated_langevin_raw_moment not implemented in _core module"
            )

        return _core.subordinated_langevin_raw_moment(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            duration,
            order,
            particles,
            time_step,
        )

    def central_moment(
        self, duration: float, order: int, particles: int, time_step: float
    ) -> float:
        """
        Calculate the central moment of the Subordinated Langevin process

        Parameters
        ----------
        duration : float
            Simulation duration
        order : int
            Order of the moment
        particles : int
            Number of particles to simulate
        time_step : float
            Time step size

        Returns
        -------
        float
            Central moment
        """
        if not hasattr(_core, "subordinated_langevin_central_moment"):
            raise NotImplementedError(
                "subordinated_langevin_central_moment not implemented in _core module"
            )

        return _core.subordinated_langevin_central_moment(
            self.drift_func,
            self.diffusion_func,
            self.start_position,
            duration,
            order,
            particles,
            time_step,
        )

    def mean(self, duration: float, particles: int, time_step: float) -> float:
        """
        Calculate the mean of the Subordinated Langevin process

        Parameters
        ----------
        duration : float
            Simulation duration
        particles : int
            Number of particles to simulate
        time_step : float
            Time step size

        Returns
        -------
        float
            Mean value
        """
        return self.raw_moment(duration, 1, particles, time_step)

    def variance(self, duration: float, particles: int, time_step: float) -> float:
        """
        Calculate the variance of the Subordinated Langevin process

        Parameters
        ----------
        duration : float
            Simulation duration
        particles : int
            Number of particles to simulate
        time_step : float
            Time step size

        Returns
        -------
        float
            Variance
        """
        return self.central_moment(duration, 2, particles, time_step)
