import BenchmarkTools: @benchmark, prettytime
import Statistics: mean, std

export bench

function bench(name::AbstractString, func::F) where F
    trial = @benchmark $func()
    times = trial.times
    stats = (
        min=minimum(times),
        max=maximum(times),
        mean=mean(times),
        std=std(times),
    )

    println(name)
    println("mean: $(prettytime(stats.mean)), min: $(prettytime(stats.min)), max: $(prettytime(stats.max)), std: $(prettytime(stats.std))")
end
