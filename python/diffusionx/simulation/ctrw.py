from diffusionx import _core
from typing import Union
from .basic import StochasticProcess, Trajectory
from .utils import check_transform
import numpy as np


real = Union[float, int]


class CTRW(StochasticProcess):
    def __init__(
        self,
        alpha: real = 1.0,
        beta: real = 2.0,
        start_position: real = 0.0,
    ):
        """
        Continuous Time Random Walk

        Args:
            alpha (real, optional): The exponent of the waiting time distribution, between 0 and 1. When alpha = 1, the waiting time is exponential, otherwise it is power-law, with tail index alpha. Default is 1.0.
            beta (real, optional): The exponent of the jump length distribution, between 0 and 2. When beta = 2, the jump length is normal, otherwise it is power-law, with tail index beta. Default is 2.0.
            start_position (real, optional): The starting position of the continuous time random walk. Default is 0.0.

        Raises:
            ValueError: If alpha is not in the range (0, 1].
            ValueError: If beta is not in the range (0, 2].
            ValueError: If the value is not a number.

        Returns:
            CTRW: The continuous time random walk object.
        """
        alpha = check_transform(alpha)
        beta = check_transform(beta)
        start_position = check_transform(start_position)

        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must be in the range (0, 1]")
        if beta <= 0 or beta > 2:
            raise ValueError("beta must be in the range (0, 2]")

        self.alpha = alpha
        self.beta = beta
        self.start_position = start_position

    def __call__(self, duration: real) -> Trajectory:
        return Trajectory(self, duration)

    def simulate(
        self, duration: real, step_size: real = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the continuous time random walk.

        Args:
            duration (real): The duration of the simulation.
            step_size (real, optional): The step size, which is not used in this simulation, but is kept for consistency with other stochastic process interfaces. Default is 0.01.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the time and position of the continuous time random walk.
        """
        duration = check_transform(duration)
        if duration <= 0:
            raise ValueError("duration must be positive, got {}".format(duration))
        return _core.ctrw_simulate_duration(
            self.alpha,
            self.beta,
            self.start_position,
            duration,
        )

    def simulate_with_step(self, num_step: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the continuous time random walk with a specified number of steps.

        Args:
            num_step (int): The number of steps in the simulation.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the time and position of the continuous time random walk.
        """
        if not isinstance(num_step, int):
            raise ValueError("num_step must be an integer")
        if num_step <= 0:
            raise ValueError("num_step must be positive")
        return _core.ctrw_simulate_step(
            self.alpha,
            self.beta,
            self.start_position,
            num_step,
        )

    def fpt(
        self,
        domain: tuple[real, real],
        max_duration: real = 1000,
    ):
        """
        Calculate the first passage time of the continuous time random walk.

        Args:
            domain (tuple[real, real]): The domain of the continuous time random walk.
            max_duration (real, optional): The maximum duration. Default is 1000.

        Returns:
            real: The first passage time of the continuous time random walk.
        """
        a = check_transform(domain[0])
        b = check_transform(domain[1])
        if a >= b:
            raise ValueError("domain must be a valid interval")
        max_duration = check_transform(max_duration)
        if max_duration <= 0:
            raise ValueError("max_duration must be positive")
        return _core.ctrw_fpt(
            self.alpha,
            self.beta,
            self.start_position,
            (a, b),
            max_duration,
        )

    def raw_moment(self, duration: real, order: int, particles: int) -> float:
        """
        Calculate the raw moment of the continuous time random walk.

        Args:
            duration (real): The duration of the simulation.
            order (int): The order of the moment.
            particles (int): The number of particles.

        Returns:
            real: The raw moment of the continuous time random walk.
        """
        if not isinstance(order, int):
            raise ValueError("order must be an integer")
        elif order < 0:
            raise ValueError("order must be non-negative")
        elif order == 0:
            return 1
        if not isinstance(particles, int):
            raise ValueError("particles must be an integer")
        elif particles <= 0:
            raise ValueError("particles must be positive")
        duration = check_transform(duration)
        if duration <= 0:
            raise ValueError("duration must be positive")
        return _core.ctrw_raw_moment(
            self.alpha,
            self.beta,
            self.start_position,
            duration,
            order,
            particles,
        )

    def central_moment(self, duration: real, order: int, particles: int):
        """
        Calculate the central moment of the continuous time random walk.

        Args:
            duration (real): The duration of the simulation.
            order (int): The order of the moment.
            particles (int): The number of particles.

        Raises:
            ValueError: If the order is not an integer.
            ValueError: If the order is negative.
            ValueError: If the order is zero.
            ValueError: If the particles number is not an integer.
            ValueError: If the particles number is not positive.

        Returns:
            real: The central moment of the continuous time random walk.
        """
        if not isinstance(order, int):
            raise ValueError("order must be an integer")
        elif order < 0:
            raise ValueError("order must be non-negative")
        elif order == 0:
            return 1
        if not isinstance(particles, int):
            raise ValueError("particles must be an integer")
        elif particles <= 0:
            raise ValueError("particles must be positive")
        duration = check_transform(duration)
        if duration <= 0:
            raise ValueError("duration must be positive")
        return _core.ctrw_central_moment(
            self.alpha,
            self.beta,
            self.start_position,
            duration,
            order,
            particles,
        )

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
    ):
        """
        Calculate the occupation time of the continuous time random walk.

        Args:
            domain (tuple[real, real]): The domain of the continuous time random walk.
            duration (real): The duration of the continuous time random walk.

        Returns:
            real: The occupation time of the continuous time random walk.
        """
        duration = check_transform(duration)
        if duration <= 0:
            raise ValueError("duration must be positive")
        a = check_transform(domain[0])
        b = check_transform(domain[1])
        if a >= b:
            raise ValueError("domain must be a valid interval")
        return _core.ctrw_occupation_time(
            self.alpha,
            self.beta,
            self.start_position,
            (a, b),
            duration,
        )

    def mean(self, duration: real, particles: int) -> float:
        """
        Calculate the mean of the continuous time random walk.

        Args:
            duration (real): The duration of the simulation.
            particles (int): The number of particles.

        Returns:
            float: The mean of the continuous time random walk.
        """
        return self.raw_moment(duration, 1, particles)

    def msd(self, duration: real, particles: int) -> float:
        """
        Calculate the mean square displacement of the continuous time random walk.

        Args:
            duration (real): The duration of the simulation.
            particles (int): The number of particles.

        Returns:
            float: The mean square displacement of the continuous time random walk.
        """
        return self.central_moment(duration, 2, particles)
