clc
clear
close all

T = 100:100:1000;
x0 = 0;
% D = 0.5;
tau = 0.01;
N = 10000;
f = @(x, t) (-x);
g = @(x, t) -1;
m = zeros(length(T), 1);
tic
% for k=1:length(T)
%     m(k) = bm_msd(T(k), x0, D, tau, N);
% end
for k=1:length(T)
    m(k) = langevin_msd(T(k), x0, f, g, tau, N);
end

toc