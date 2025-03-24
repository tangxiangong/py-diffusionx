function res = stable_rands(alpha, beta, sigma, mu, n)
    % 根据alpha值选择不同的生成方法
    if alpha == 1.0
        res = stable_rands_alpha_one(beta, sigma, mu, n);
    else
        res = stable_rands_alpha(alpha, beta, sigma, mu, n);
    end
end

function res = stable_rands_alpha(alpha, beta, sigma, mu, n)
    % 一次性生成所有随机数
    half_pi = pi / 2.0;
    v = (rand(n, 1) - 1/2) * 2 * half_pi;
    w = -log(rand(n, 1));  % 批量生成指数分布随机数
    
    % 向量化计算
    tmp = beta * tan(alpha * half_pi);
    b = atan(tmp) / alpha;
    s = (1.0 + tmp * tmp)^(1.0 / (2.0 * alpha));
    
    % 应用向量运算
    c1 = alpha * sin(v + b) ./ (cos(v).^(1.0 / alpha));
    c2 = cos(v - alpha * (v + b)) ./ (w.^(1.0 / alpha));
    
    % 计算结果
    standard_samples = s * ones(n, 1) .* c1 .* c2;
    res = sigma * standard_samples + mu;
end

function res = stable_rands_alpha_one(beta, sigma, mu, n)
    % 一次性生成所有随机数
    half_pi = pi / 2.0;
    v = (rand(n, 1) - 1/2) * 2 * half_pi;
    w = -log(rand(n, 1));  % 批量生成指数分布随机数
    
    % 向量化计算
    c1 = (half_pi + beta * v) .* tan(v);
    c2 = ((half_pi * w .* cos(v)) ./ log(half_pi + beta * v)) * beta;
    
    % 计算最终结果
    standard_samples = 2.0 * (c1 - c2) / pi;
    res = sigma * standard_samples + mu + 2.0 * beta * sigma * sigma * log(sigma) / pi;
end 