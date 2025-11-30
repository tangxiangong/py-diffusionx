import Statistics: mean, std, minimum, maximum
import Printf: @sprintf
import Random: rand, randn, randexp, rand

include("stable.jl")
stable_rands(0.7, 1.0, 1.0, 0.0, 10)

# 从命令行参数获取 bench_size，默认为 20
bench_size = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20

function timeit(func, bench_size)
    result = zeros(bench_size)
    for i in 1:bench_size
        result[i] = @elapsed func()
    end
    result
end

function show_timeit(result)
    mean_v = mean(result)
    stddev_v = std(result)
    min_v = minimum(result)
    max_v = maximum(result)
    println(@sprintf("mean: %.4f, stddev: %.4f, min: %.4f, max: %.4f", mean_v, stddev_v, min_v, max_v))
    println()
end

N = 10_000_000

println("=========================Julia=========================")
println()

println("bench size: $bench_size, length of random vectors: $N")
println("unit: second")
println()

println("-------------uniform random number sampling------------")
uniform = () -> rand(N)
show_timeit(timeit(uniform, bench_size))

println("-------------normal random number sampling-------------")
normal = () -> randn(N)
show_timeit(timeit(normal, bench_size))

println("-------------stable random number sampling-------------")
stable = () -> stable_rands(0.7, 0.0, 1.0, 0.0, N)
show_timeit(timeit(stable, bench_size))

println("=======================================================")
println()
