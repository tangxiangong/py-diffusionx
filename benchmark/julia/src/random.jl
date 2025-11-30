import Random: rand, randn, randexp, rand

include("stable.jl")
include("utils.jl")

stable_rands(0.7, 1.0, 1.0, 0.0, 10)

N = 10_000_000

println("=========================Julia=========================")
println()

bench("uniform random number sampling", () -> rand(N))
bench("normal random number sampling", () -> randn(N))
bench("exponential random number sampling", () -> randexp(N))
bench("stable random number sampling", () -> stable_rands(0.7, 0.0, 1.0, 0.0, N))
