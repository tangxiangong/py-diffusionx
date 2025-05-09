import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional, Union, Callable, Literal, Any, cast
from .basic import StochasticProcess
from dataclasses import dataclass, field

# Type aliases
real = Union[float, int]
PathData = tuple[np.ndarray, np.ndarray]
PlotType = Literal[
    "trajectory", "multiple", "statistics", "msd", "histogram", "density"
]


@dataclass
class PlotConfig:
    """Configuration class for stochastic process visualization"""

    # Basic configuration
    plot_type: PlotType = "trajectory"
    title: str = "Stochastic Process Visualization"
    xlabel: str = "Time"
    ylabel: str = "Position"
    color: Optional[Union[str, list[str]]] = None
    figsize: tuple[int, int] = (10, 6)
    grid: bool = True
    save_path: Optional[str] = None
    show: bool = True

    # Line style and marker configuration
    linestyle: Optional[Union[str, list[str]]] = None
    linewidth: Optional[Union[float, list[float]]] = None
    marker: Optional[Union[str, list[str]]] = None
    markersize: Optional[Union[float, list[float]]] = None
    alpha: Optional[Union[float, list[float]]] = None

    # Stochastic process simulation configuration
    duration: real = 10.0
    step_size: real = 0.01
    n_trajectories: int = 100

    # Statistical properties configuration
    statistics: str = "mean"
    confidence_interval: Optional[float] = 0.95

    # MSD configuration
    theoretical_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
    log_scale: bool = False

    # Histogram configuration
    time_point: Optional[Union[int, real]] = None
    bins: int = 50
    kde: bool = True

    # Density plot configuration
    n_time_points: int = 5

    # Other configuration
    labels: list[str] = field(default_factory=list)
    legend_loc: str = "best"
    ci_color: str = "lightblue"
    theory_color: str = "red"
    kde_color: str = "darkblue"
    cmap: str = "viridis"

    # Custom axis configuration
    ax: Optional[plt.Axes] = None
    show_with_custom_ax: bool = False

    # Step plot configuration
    draw_as_steps: bool = False
    step_style: Optional[Literal["pre", "post", "mid"]] = None

    def __post_init__(self):
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.n_trajectories <= 0:
            raise ValueError("n_trajectories must be positive")
        if self.confidence_interval is not None and not (
            0 < self.confidence_interval < 1
        ):
            raise ValueError("confidence_interval must be between 0 and 1 (exclusive)")
        if self.bins <= 0:
            raise ValueError("bins for histogram must be positive")
        if self.n_time_points <= 0:
            raise ValueError("n_time_points for density plot must be positive")

        # Validate statistics choice
        valid_stats = ["mean", "variance", "std"]
        if self.statistics not in valid_stats:
            raise ValueError(
                f"Unsupported statistics type: '{self.statistics}'. Must be one of {valid_stats}"
            )

    def get_style_at_index(
        self, style_prop: Optional[Union[Any, list[Any]]], idx: int, default: Any
    ) -> Any:
        """Get the style property at the specified index, or return the property itself if not a list"""
        if style_prop is None:
            return default
        elif isinstance(style_prop, list):
            return style_prop[idx % len(style_prop)]
        else:
            return style_prop


def _plot_trajectory(data: PathData, config: PlotConfig) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a single trajectory.

    Args:
        data: Trajectory data (tuple of time and position arrays)
        config: Plot configuration object

    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    times, positions = data

    if config.ax is None:
        fig, ax = plt.subplots(figsize=config.figsize)
    else:
        fig = cast(plt.Figure, config.ax.figure)
        ax = config.ax

    ax.plot(
        times,
        positions,
        color=config.color or "blue",
        linestyle=config.get_style_at_index(config.linestyle, 0, "-"),
        linewidth=config.get_style_at_index(config.linewidth, 0, 1.0),
        marker=config.get_style_at_index(config.marker, 0, None),
        markersize=config.get_style_at_index(config.markersize, 0, 6),
        alpha=config.get_style_at_index(config.alpha, 0, 1.0),
    )

    return fig, ax


def _plot_multiple_trajectories(
    data: List[PathData], config: PlotConfig
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot multiple trajectories for comparison.

    Args:
        data: List of trajectory data
        config: Plot configuration object

    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=config.figsize)

    if config.color is None:
        # Use default color cycle
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    elif isinstance(config.color, str):
        colors = [config.color]
    else:
        colors = config.color

    labels = config.labels or [f"Trajectory {i + 1}" for i in range(len(data))]

    for i, path_data in enumerate(data):
        times, positions = path_data
        ax.plot(
            times,
            positions,
            color=colors[i % len(colors)],
            linestyle=config.get_style_at_index(config.linestyle, i, "-"),
            linewidth=config.get_style_at_index(config.linewidth, i, 1.0),
            marker=config.get_style_at_index(config.marker, i, None),
            markersize=config.get_style_at_index(config.markersize, i, 6),
            alpha=config.get_style_at_index(config.alpha, i, 0.8),
            label=labels[i],
        )
    ax.legend(loc=config.legend_loc)

    return fig, ax


def _generate_trajectories(
    process: StochasticProcess, config: PlotConfig
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Generate multiple trajectories for statistical analysis.

    Args:
        process: Stochastic process object
        config: Plot configuration object

    Returns:
        times: Time array
        all_trajectories: List of all trajectories
        all_trajectories_array: Numpy array of all trajectories
    """
    all_trajectories = []
    times = None

    for _ in range(config.n_trajectories):
        t, pos = process.simulate(config.duration, config.step_size)
        all_trajectories.append(pos)
        if times is None:
            times = t

    # Ensure times is not None
    if times is None:
        raise ValueError("Could not generate valid trajectory data")

    all_trajectories_array = np.array(all_trajectories)
    return times, all_trajectories, all_trajectories_array


def _plot_statistics(
    process: StochasticProcess, config: PlotConfig
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot statistical properties (mean, variance, or standard deviation).

    Args:
        process: Stochastic process object
        config: Plot configuration object

    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    times, _, all_trajectories_array = _generate_trajectories(process, config)

    # Calculate specified statistics
    if config.statistics == "mean":
        stat_value = np.mean(all_trajectories_array, axis=0)
        if config.ylabel == "Position":
            config.ylabel = "Average Position"
    elif config.statistics == "variance":
        stat_value = np.var(all_trajectories_array, axis=0)
        if config.ylabel == "Position":
            config.ylabel = "Position Variance"
    elif config.statistics == "std":
        stat_value = np.std(all_trajectories_array, axis=0)
        if config.ylabel == "Position":
            config.ylabel = "Position Standard Deviation"
    else:
        raise ValueError(f"Unsupported statistic type: {config.statistics}")

    fig, ax = plt.subplots(figsize=config.figsize)
    if times is not None:
        ax.plot(
            times,
            stat_value,
            color=config.color or "blue",
            linestyle=config.get_style_at_index(config.linestyle, 0, "-"),
            linewidth=config.get_style_at_index(config.linewidth, 0, 2.0),
            marker=config.get_style_at_index(config.marker, 0, None),
            markersize=config.get_style_at_index(config.markersize, 0, 6),
            label=config.statistics,
        )

        if config.confidence_interval is not None:
            alpha = 1 - config.confidence_interval
            lower_percentile = alpha / 2 * 100
            upper_percentile = (1 - alpha / 2) * 100

            lower_bound = np.percentile(
                all_trajectories_array, lower_percentile, axis=0
            )
            upper_bound = np.percentile(
                all_trajectories_array, upper_percentile, axis=0
            )

            ax.fill_between(
                times,
                lower_bound,
                upper_bound,
                color=config.ci_color,
                alpha=0.5,
                label=f"{config.confidence_interval * 100:.0f}% Confidence Interval",
            )
    ax.legend(loc="best")

    return fig, ax


def _plot_msd(
    process: StochasticProcess, config: PlotConfig
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot mean square displacement.

    Args:
        process: Stochastic process object
        config: Plot configuration object

    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    times, all_trajectories, _ = _generate_trajectories(process, config)

    # Ensure each trajectory starts from 0
    shifted_trajectories = [traj - traj[0] for traj in all_trajectories]
    all_trajectories_array = np.array(shifted_trajectories)
    msd = np.mean(np.square(all_trajectories_array), axis=0)

    fig, ax = plt.subplots(figsize=config.figsize)
    ax.plot(
        times,
        msd,
        color=config.color or "blue",
        linestyle=config.get_style_at_index(config.linestyle, 0, "-"),
        linewidth=config.get_style_at_index(config.linewidth, 0, 2.0),
        marker=config.get_style_at_index(config.marker, 0, None),
        markersize=config.get_style_at_index(config.markersize, 0, 6),
        label="Simulated MSD",
    )

    if config.theoretical_func is not None:
        theory_values = config.theoretical_func(times)
        ax.plot(
            times,
            theory_values,
            color=config.theory_color,
            linestyle="--",
            linewidth=2.0,
            label="Theoretical MSD",
        )

    if config.log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.legend(loc="best")

    return fig, ax


def _plot_histogram(
    process: StochasticProcess, config: PlotConfig
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot position distribution histogram.

    Args:
        process: Stochastic process object
        config: Plot configuration object

    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    # Generate trajectory data
    times, _, _ = _generate_trajectories(process, config)

    # Determine time index
    time_index = _determine_time_index(times, config.time_point)

    # Collect position data at specified time
    positions_at_time = _collect_positions_at_time(process, config, time_index)

    # Create figure
    fig, ax = plt.subplots(figsize=config.figsize)

    # Plot histogram
    _, _, _ = ax.hist(
        positions_at_time,
        bins=config.bins,
        density=True,
        color=config.color or "skyblue",
        alpha=config.get_style_at_index(config.alpha, 0, 0.7),
        label="Histogram",
    )

    # Add KDE curve
    if config.kde:
        _add_kde_to_histogram(ax, positions_at_time, config)

    # Set title
    actual_time = times[time_index]
    title = f"{config.title} (t = {actual_time:.2f})"
    ax.legend(loc="best")
    config.title = title

    return fig, ax


def _determine_time_index(
    times: np.ndarray, time_point: Optional[Union[int, real]]
) -> int:
    """
    Determine time index for histogram.

    Args:
        times: Time array
        time_point: Specified time point or index

    Returns:
        int: Time index
    """
    if isinstance(time_point, int):
        return time_point
    elif isinstance(time_point, (float, int)):
        return np.abs(times - time_point).argmin()
    else:
        # Default to the last time point
        return -1


def _collect_positions_at_time(
    process: StochasticProcess, config: PlotConfig, time_index: int
) -> List[float]:
    """
    Collect position data at specified time point.

    Args:
        process: Stochastic process object
        config: Plot configuration object
        time_index: Time index

    Returns:
        List[float]: List of position data
    """
    positions_at_time = []

    for _ in range(config.n_trajectories):
        _, positions = process.simulate(config.duration, config.step_size)
        positions_at_time.append(positions[time_index])

    return positions_at_time


def _add_kde_to_histogram(
    ax: plt.Axes, positions: List[float], config: PlotConfig
) -> None:
    """
    Add kernel density estimation curve to histogram.

    Args:
        ax: Matplotlib axes object
        positions: Position data
        config: Plot configuration object
    """
    try:
        from scipy.stats import gaussian_kde  # type: ignore

        kde_model = gaussian_kde(positions)
        x_range = np.linspace(np.min(positions), np.max(positions), 1000)
        ax.plot(
            x_range,
            kde_model(x_range),
            color=config.kde_color,
            linewidth=config.get_style_at_index(config.linewidth, 0, 2.0),
            label="Kernel Density Estimation",
        )
    except ImportError:
        print("scipy library is required for KDE plotting")


def _plot_density(process: StochasticProcess, config: PlotConfig) -> plt.Figure:
    """
    Plot density distribution evolution over time.

    Args:
        process: Stochastic process object
        config: Plot configuration object

    Returns:
        fig: Matplotlib figure object
    """
    all_trajectories = []
    times = None

    for _ in range(config.n_trajectories):
        t, pos = process.simulate(config.duration, config.step_size)
        all_trajectories.append(pos)
        if times is None:
            times = t

    if times is None:
        raise ValueError("Could not generate sufficient trajectory data")

    time_indices = np.linspace(0, len(times) - 1, config.n_time_points).astype(int)

    fig, axes = plt.subplots(
        config.n_time_points, 1, figsize=config.figsize, sharex=True
    )
    if config.n_time_points == 1:
        axes = [axes]

    cmap_colors = plt.cm.get_cmap(config.cmap, config.n_time_points)

    for i, time_idx in enumerate(time_indices):
        positions = [traj[time_idx] for traj in all_trajectories]

        axes[i].hist(
            positions,
            bins=config.bins,
            density=True,
            color=cmap_colors(i / config.n_time_points),
            alpha=config.get_style_at_index(config.alpha, i, 0.7),
        )

        axes[i].text(
            0.02,
            0.85,
            f"t = {times[time_idx]:.2f}",
            transform=axes[i].transAxes,
            fontsize=10,
        )

        axes[i].set_ylabel("Probability Density")

        if i == config.n_time_points - 1:
            axes[i].set_xlabel(config.xlabel)

    plt.tight_layout()
    fig.suptitle(config.title, y=1.02)

    return fig


def _apply_common_settings(fig: plt.Figure, ax: plt.Axes, config: PlotConfig) -> None:
    """
    Apply common plot settings.

    Args:
        fig: Matplotlib figure object
        ax: Matplotlib axes object
        config: Plot configuration object
    """
    ax.set_title(config.title)
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)

    if config.grid:
        ax.grid(True, linestyle="--", alpha=0.7)

    if config.save_path:
        plt.savefig(config.save_path, dpi=300, bbox_inches="tight")

    # Control whether to show the figure
    if (
        config.show
        and config.plot_type == "trajectory"
        and isinstance(config.ax, plt.Axes)
        and config.ax is not None
    ):
        # Only need explicit control of display for trajectory type with custom ax
        if config.show_with_custom_ax:
            plt.show()
    elif config.show:
        plt.show()


def plot(
    data: Union[PathData, list[PathData], StochasticProcess],
    config: Optional[PlotConfig] = None,
    **kwargs,
) -> plt.Figure:
    """Main plotting function for stochastic processes and trajectories."""

    _config: PlotConfig
    if config is None:
        _config = PlotConfig(**kwargs)  # Validated by __post_init__
    else:
        if kwargs:  # User provided a base config and overrides via kwargs
            config_dict = (
                config.__dict__.copy()
            )  # Use .copy() to avoid modifying original if it's reused
            config_dict.update(kwargs)
            _config = PlotConfig(**config_dict)  # Re-validate with __post_init__
        else:
            _config = config  # Already an instance, assume validated if user constructed it properly

    # Auto-determine plot_type if it's the default and data type suggests a better one
    # Or validate if user-set plot_type is compatible with data.
    original_user_plot_type = (
        kwargs.get("plot_type", None) if config is None else config.plot_type
    )
    is_plot_type_default_or_from_kwargs = config is None or (
        _config.plot_type == PlotConfig().plot_type and not original_user_plot_type
    )

    if isinstance(data, StochasticProcess):
        # For a StochasticProcess, all plot types are potentially valid as trajectories can be generated.
        # If plot_type was not explicitly set by user (or was default), and it's something like "multiple",
        # it might be ambiguous. Defaulting to "trajectory" if not specified seems reasonable.
        if is_plot_type_default_or_from_kwargs and _config.plot_type not in [
            "trajectory",
            "statistics",
            "msd",
            "histogram",
            "density",
        ]:
            _config.plot_type = "trajectory"  # Sensible default for a process

    elif (
        isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], np.ndarray)
    ):  # PathData
        if _config.plot_type not in ["trajectory"]:
            if (
                original_user_plot_type == _config.plot_type
            ):  # User explicitly set an incompatible type
                raise ValueError(
                    f"Plot type '{_config.plot_type}' is not suitable for single trajectory data (PathData). "
                    f"Valid type is 'trajectory'."
                )
            # If plot_type was default or inferred incorrectly, switch to trajectory
            _config.plot_type = "trajectory"

    elif isinstance(data, list) and all(
        isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], np.ndarray)
        for item in data
    ):
        # List[PathData]
        if _config.plot_type not in ["multiple"]:
            if original_user_plot_type == _config.plot_type:
                raise ValueError(
                    f"Plot type '{_config.plot_type}' is not suitable for a list of trajectories (List[PathData]). "
                    f"Valid type is 'multiple'."
                )
            _config.plot_type = "multiple"
    else:
        raise TypeError(
            "Invalid data type for plotting. Expected StochasticProcess, PathData (tuple[np.ndarray, np.ndarray]), "
            "or List[PathData]."
        )

    # Dispatch to actual plot functions
    fig: plt.Figure
    ax: plt.Axes

    # Handle ax from config
    # If an Axes object is provided in the config, use it directly.
    # Otherwise, the individual _plot_* functions will create a new figure and axes.
    # This logic implies _plot_* functions need to handle config.ax being None or an Axes object.

    if _config.plot_type == "trajectory":
        if isinstance(data, StochasticProcess):
            # Simulate a single trajectory if a process is given
            if (
                _config.duration is None
            ):  # Should not happen due to PlotConfig default and __post_init__
                raise ValueError(
                    "Duration must be specified in PlotConfig for trajectory plot from StochasticProcess"
                )
            # The simulate method from basic.StochasticProcess is (self, duration, step_size)
            # We rely on the specific process's simulate method for actual simulation.
            path_data = data.simulate(_config.duration, _config.step_size)
            fig, ax = _plot_trajectory(path_data, _config)
        elif isinstance(data, tuple):  # PathData
            fig, ax = _plot_trajectory(data, _config)
        else:
            raise ValueError(
                "For 'trajectory' plot, data must be StochasticProcess or PathData."
            )

    elif _config.plot_type == "multiple":
        if isinstance(data, list):  # List[PathData]
            fig, ax = _plot_multiple_trajectories(data, _config)
        elif isinstance(data, StochasticProcess):
            # Generate N trajectories for 'multiple' plot type from a process
            trajectories_data = []
            for _ in range(_config.n_trajectories if _config.n_trajectories > 0 else 1):
                path_data = data.simulate(_config.duration, _config.step_size)
                trajectories_data.append(path_data)
            fig, ax = _plot_multiple_trajectories(trajectories_data, _config)
        else:
            raise ValueError(
                "For 'multiple' plot, data must be List[PathData] or StochasticProcess."
            )

    elif _config.plot_type == "statistics":
        if not isinstance(data, StochasticProcess):
            raise ValueError("For 'statistics' plot, data must be a StochasticProcess.")
        fig, ax = _plot_statistics(data, _config)

    elif _config.plot_type == "msd":
        if not isinstance(data, StochasticProcess):
            raise ValueError("For 'msd' plot, data must be a StochasticProcess.")
        fig, ax = _plot_msd(data, _config)

    elif _config.plot_type == "histogram":
        if not isinstance(data, StochasticProcess):
            raise ValueError("For 'histogram' plot, data must be a StochasticProcess.")
        fig, ax = _plot_histogram(data, _config)

    elif _config.plot_type == "density":
        if not isinstance(data, StochasticProcess):
            raise ValueError("For 'density' plot, data must be a StochasticProcess.")
        # _plot_density returns fig directly
        fig = _plot_density(data, _config)
        # For density plots, ax might be an array or specific. _apply_common_settings might need adjustment
        # or _plot_density handles its own common settings if it's too different.
        # For now, assume it returns a figure and _apply_common_settings works with its main ax or is skipped.
        if hasattr(fig, "axes") and len(fig.axes) > 0:
            _apply_common_settings(
                fig, fig.axes[0], _config
            )  # Apply to the first axes for simplicity
        # else: might need more specific handling if fig from _plot_density has no obvious single ax

    else:
        raise ValueError(f"Unsupported plot_type: '{_config.plot_type}'")

    # Apply common settings if not a density plot that handles its own, or if ax is available
    if _config.plot_type != "density" and "ax" in locals():  # ensure ax is defined
        _apply_common_settings(fig, ax, _config)

    if _config.save_path:
        fig.savefig(_config.save_path)

    # Handle showing the plot
    if _config.ax is None and _config.show:
        plt.show()
    elif _config.ax is not None and _config.show_with_custom_ax and _config.show:
        plt.show()  # User explicitly wants to show even if custom ax was used

    plt.close(fig)  # Close the figure to free memory, esp. in loops or scripts

    return fig
