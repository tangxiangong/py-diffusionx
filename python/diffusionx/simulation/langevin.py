from typing import Union, Callable, Optional, Tuple
import numpy as np
from diffusionx import _core
from .basic import StochasticProcess, Trajectory
from .utils import check_transform


real = Union[float, int]
DriftFunc = Callable[[float, float], float]
DiffusionFunc = Callable[[float, float], float]


class Langevin(StochasticProcess):
    def __init__(
        self,
        drift_func: DriftFunc,
        diffusion_func: DiffusionFunc,
        start_position: real = 0.0,
    ):
        """
        Initialize a Langevin equation object.

        dx(t) = f(x(t), t) dt + g(x(t), t) dW(t), x(0) = x0

        where W(t) is the Weiner process or called Brownian motion.

        Args:
            drift_func (DriftFunc): The drift function of the Langevin equation, f(x, t).
            diffusion_func (DiffusionFunc): The diffusion function of the Langevin equation, g(x, t).
            start_position (real, optional): Starting position of the Langevin equation. Defaults to 0.0.

        Raises:
            ValueError: If value is not a number.

        Returns:
            Langevin: A Langevin equation object.
        """
        start_position = check_transform(start_position)

        self.drift_func = drift_func
        self.diffusion_func = diffusion_func
        self.start_position = start_position

    def __call__(self, duration: real) -> Trajectory:
        return Trajectory(self, duration)

    def simulate(
        self, duration: real, step_size: real = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the Langevin equation.

        Args:
            duration (real): Duration of the simulation.
            step_size (real, optional): Step size of the simulation. Defaults to 0.01.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Time and position arrays.
        """
        duration = check_transform(duration)
        step_size = check_transform(step_size)
        if duration <= 0:
            raise ValueError("duration must be positive")
        if step_size <= 0:
            raise ValueError("step_size must be positive")

        num = int(np.ceil(duration / step_size))
        t = np.linspace(0, num * step_size, num + 1)
        x = np.zeros(num + 1)
        x[0] = self.start_position
        noise = np.random.normal(0, 1, num)

        for i in range(1, num + 1):
            x[i] = (
                x[i - 1]
                + self.drift_func(x[i - 1], t[i - 1]) * step_size
                + self.diffusion_func(x[i - 1], t[i - 1])
                * noise[i - 1]
                * np.sqrt(step_size)
            )

        return t, x

    def fpt(
        self,
        domain: Tuple[real, real],
        step_size: real = 0.01,
        max_duration: real = 1000,
    ) -> Optional[float]:
        """
        Calculate the first passage time of the Langevin equation.

        Args:
            domain (Tuple[real, real]): Domain of the first passage time.
            step_size (real, optional): Step size of the simulation. Defaults to 0.01.
            max_duration (real, optional): Maximum duration of the simulation. Defaults to 1000.

        Returns:
            Optional[float]: First passage time, or None if not found.
        """
        lower, upper = domain
        lower = check_transform(lower)
        upper = check_transform(upper)
        step_size = check_transform(step_size)
        max_duration = check_transform(max_duration)

        if lower >= upper:
            raise ValueError("lower bound must be less than upper bound")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if max_duration <= 0:
            raise ValueError("max_duration must be positive")

        if not (lower <= self.start_position <= upper):
            return 0.0

        t, x = self.simulate(max_duration, step_size)
        for i in range(len(t)):
            if x[i] <= lower or x[i] >= upper:
                return t[i]

        return None

    def raw_moment(
        self, duration: real, order: int, particles: int, step_size: float = 0.01
    ) -> float:
        """
        Calculate the raw moment of the Langevin equation.

        Args:
            duration (real): Duration of the simulation.
            order (int): Order of the moment.
            particles (int): Number of particles.
            step_size (float, optional): Step size of the simulation. Defaults to 0.01.

        Returns:
            float: Raw moment.
        """
        duration = check_transform(duration)
        step_size = check_transform(step_size)

        if duration <= 0:
            raise ValueError("duration must be positive")
        if order <= 0:
            raise ValueError("order must be positive")
        if particles <= 0:
            raise ValueError("particles must be positive")
        if step_size <= 0:
            raise ValueError("step_size must be positive")

        result = 0.0
        for _ in range(particles):
            _, x = self.simulate(duration, step_size)
            result += x[-1] ** order

        return result / particles

    def central_moment(
        self, duration: real, order: int, particles: int, step_size: float = 0.01
    ) -> float:
        """
        Calculate the central moment of the Langevin equation.

        Args:
            duration (real): Duration of the simulation.
            order (int): Order of the moment.
            particles (int): Number of particles.
            step_size (float, optional): Step size of the simulation. Defaults to 0.01.

        Returns:
            float: Central moment.
        """
        duration = check_transform(duration)
        step_size = check_transform(step_size)

        if duration <= 0:
            raise ValueError("duration must be positive")
        if order <= 0:
            raise ValueError("order must be positive")
        if particles <= 0:
            raise ValueError("particles must be positive")
        if step_size <= 0:
            raise ValueError("step_size must be positive")

        # Calculate mean
        mean = 0.0
        for _ in range(particles):
            _, x = self.simulate(duration, step_size)
            mean += x[-1]
        mean /= particles

        # Calculate central moment
        result = 0.0
        for _ in range(particles):
            _, x = self.simulate(duration, step_size)
            result += (x[-1] - mean) ** order

        return result / particles

    def occupation_time(
        self,
        domain: Tuple[real, real],
        duration: real,
        step_size: real = 0.01,
    ) -> float:
        """
        Calculate the occupation time of the Langevin equation.

        Args:
            domain (Tuple[real, real]): Domain of the occupation time.
            duration (real): Duration of the simulation.
            step_size (real, optional): Step size of the simulation. Defaults to 0.01.

        Returns:
            float: Occupation time.
        """
        lower, upper = domain
        lower = check_transform(lower)
        upper = check_transform(upper)
        duration = check_transform(duration)
        step_size = check_transform(step_size)

        if lower >= upper:
            raise ValueError("lower bound must be less than upper bound")
        if duration <= 0:
            raise ValueError("duration must be positive")
        if step_size <= 0:
            raise ValueError("step_size must be positive")

        t, x = self.simulate(duration, step_size)
        occupation_time = 0.0

        for i in range(1, len(t)):
            if lower <= x[i] <= upper:
                occupation_time += t[i] - t[i - 1]

        return occupation_time
