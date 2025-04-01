import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional, Union, Callable, Literal, Any, cast
from .basic import StochasticProcess
from dataclasses import dataclass, field

# Type aliases
real = Union[float, int]
PathData = Tuple[np.ndarray, np.ndarray]
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
    color: Optional[Union[str, List[str]]] = None
    figsize: Tuple[int, int] = (10, 6)
    grid: bool = True
    save_path: Optional[str] = None
    show: bool = True

    # Line style and marker configuration
    linestyle: Optional[Union[str, List[str]]] = None
    linewidth: Optional[Union[float, List[float]]] = None
    marker: Optional[Union[str, List[str]]] = None
    markersize: Optional[Union[float, List[float]]] = None
    alpha: Optional[Union[float, List[float]]] = None

    # Stochastic process simulation configuration
    duration: Optional[real] = None
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
    labels: List[str] = field(default_factory=list)
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

    def get_style_at_index(
        self, style_prop: Optional[Union[Any, List[Any]]], idx: int, default: Any
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
        t, pos = process.simulate(config.duration, config.step_size)  # type: ignore
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
        _, positions = process.simulate(config.duration, config.step_size)  # type: ignore
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
        t, pos = process.simulate(config.duration, config.step_size)  # type: ignore
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
        positions = [traj[time_idx] for traj in all_trajectories]  # type: ignore

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
    data: Union[PathData, List[PathData], StochasticProcess],
    config: Optional[PlotConfig] = None,
    **kwargs,
) -> plt.Figure:
    """
    General visualization function for stochastic processes.

    Args:
        data: Data source (path data tuple, path data list, or stochastic process object)
        config: Visualization configuration object, creates default if None
        **kwargs: Parameters that can override configuration object parameters

    Returns:
        plt.Figure: Figure object
    """
    # Create default configuration if none provided
    if config is None:
        config = PlotConfig()

    # Override configuration with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    ax = None

    # Select appropriate plotting function based on data type and plot type
    if config.plot_type == "trajectory" and isinstance(data, tuple):
        # Single trajectory visualization
        fig, ax = _plot_trajectory(data, config)

    elif config.plot_type == "multiple" and isinstance(data, list):
        # Multiple trajectory comparison
        fig, ax = _plot_multiple_trajectories(data, config)

    elif config.plot_type in [
        "statistics",
        "msd",
        "histogram",
        "density",
    ] and isinstance(data, StochasticProcess):
        # Check required duration parameter
        if config.duration is None:
            raise ValueError(
                "duration parameter is required when using StochasticProcess object"
            )

        # Choose appropriate function based on plot type
        if config.plot_type == "statistics":
            fig, ax = _plot_statistics(data, config)

        elif config.plot_type == "msd":
            fig, ax = _plot_msd(data, config)

        elif config.plot_type == "histogram":
            fig, ax = _plot_histogram(data, config)

        elif config.plot_type == "density":
            fig = _plot_density(data, config)
            # Density plot handles saving and display internally, return directly
            return fig
    else:
        raise ValueError(
            f"Unsupported plot type and data combination: {config.plot_type}, {type(data)}"
        )

    # Apply common settings (density type already handled in its function)
    if ax is not None:
        _apply_common_settings(fig, ax, config)

    return fig
