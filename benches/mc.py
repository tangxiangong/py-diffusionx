import numpy as np

def simulate_bm(total_time: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    n_steps = int(total_time / dt)
    displacements = np.sqrt(2 * dt) * np.random.normal(0, 1, n_steps)
    # 初始位置为0
    positions = np.concatenate(([0], np.cumsum(displacements)))
    times = np.arange(0, (n_steps + 1) * dt, dt)
    return times, positions

def mean_bm(n_trajectories: int, total_time: float, dt: float) -> float:
    displacements = np.zeros(n_trajectories)
    for i in range(n_trajectories):
        _, positions = simulate_bm(total_time, dt)
        displacements[i] = positions[-1]
    return np.mean(displacements)

def msd_bm(n_trajectories: int, total_time: float, dt: float) -> float:
    displacements = np.zeros(n_trajectories)
    mean = mean_bm(n_trajectories, total_time, dt)
    for i in range(n_trajectories):
        _, positions = simulate_bm(total_time, dt)
        displacements[i] = (positions[-1] - mean)**2

    msd = np.mean(displacements)
    return msd
