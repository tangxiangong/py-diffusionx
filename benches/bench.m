clc; clear; close all

N = 10000000;
M = 10;

s = 0.0;
for k=1:M
    start = tic;
    stable_rands(1.7, 0, 1, 0, N);
    end_ = toc(start);
    s = s + end_;
end
t = s/M
