function v = bm_msd(T, x0, D, tau, N)
    r = 0.0;
    parfor k = 1 : N
        [~, x] = simulate(T, x0, D, tau);
        r = r + (x(end) - x(1)) ^ 2;
    end
    v = r / N;
end

function[t, x] = simulate(T, x0, D, tau)
    t = 0:tau:T;
    n = length(t) - 1;
    noise = sqrt(2 * D * tau).*randn(n, 1);
    x = cumsum([x0; noise]);
end