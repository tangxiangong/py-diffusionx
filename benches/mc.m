function [times, positions] = simulate_bm(total_time, dt)
    n_steps = floor(total_time / dt);
    displacements = sqrt(2 * dt) * randn(1, n_steps);
    % 初始位置为0
    positions = [0, cumsum(displacements)];
    times = 0:dt:(n_steps * dt);
end

function mean_val = mean_bm(n_trajectories, total_time, dt)
    displacements = zeros(1, n_trajectories);
    for i = 1:n_trajectories
        [~, positions] = simulate_bm(total_time, dt);
        displacements(i) = positions(end);
    end
    mean_val = mean(displacements);
end

function msd = msd_bm(n_trajectories, total_time, dt)
    displacements = zeros(1, n_trajectories);
    mean_val = mean_bm(n_trajectories, total_time, dt);
    parfor i = 1:n_trajectories
        [~, positions] = simulate_bm(total_time, dt);
        displacements(i) = (positions(end) - mean_val)^2;
    end
    msd = mean(displacements);
end
