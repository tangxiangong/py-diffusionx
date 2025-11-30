function v = langevin_msd(T, x0, f, g, tau, N) 
    r = 0.0;
    parfor k = 1 : N
        [~, x] = simulate(T, x0, f, g, tau);
        r = r + (x(end) - x(1)) ^ 2;
    end 
    v = r / N;
end

function[t, x] = simulate(T, x0, f, g, tau)
    t = 0:tau:T;
    n = length(t) - 1;
    noise = randn(n, 1);
    x = zeros(n+1, 1);
    x(1) = x0;
    for k=1:n 
        dw = g(x(k), t(k+1)) * sqrt(tau) * noise(k);
        x(k+1) = x(k) + f(x(k), t(k+1)) * tau + dw;
    end
end