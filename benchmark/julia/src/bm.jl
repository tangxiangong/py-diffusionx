using Random
using ThreadsX
using BenchmarkTools

struct Bm
    x₀::Float64
    D::Float64
    function Bm(x₀::Float64=0.0, D::Float64=1.0)
        new(x₀, D)
    end
end
StandardBm() = Bm(0.0, 0.5)

function simulate(bm::Bm, T::Float64, τ::Float64=0.01)
    sigma = sqrt(2 * bm.D * τ)
    t = collect(0:τ:T)
    x = Vector{Float64}(undef, length(t))
    n = length(t) - 1
    noise = randn(n - 1)
    current_x = bm.x₀
    @inbounds for i in 1:(n-1)
        noise[i] *= sigma
        current_x += noise[i]
        x[i] = current_x
    end
    last_step = T - (n - 1) * τ
    noise = randn() * sqrt(2 * bm.D * last_step)
    current_x += noise
    x[end] = current_x
    t, x
end

function displacement(bm::Bm, T::Float64, τ::Float64=0.01)
    n = ceil(Int, T / τ)
    sigma = sqrt(2 * bm.D * τ)
    noise = randn(n - 1)
    delta_x = 0.0
    @inbounds for i in 1:(n-1)
        delta_x += noise[i] * sigma
    end
    last_step = T - (n - 1) * τ
    delta_x += randn() * sqrt(2 * bm.D * last_step)
    delta_x
end


function msd(bm::Bm, T::Float64, N::Int=10_000, τ::Float64=0.01)::Float64
    displacements = ThreadsX.map(1:N) do _
        displacement(bm, T, τ)
    end
    sum(displacements .^ 2) / N
end

bm = StandardBm()
msd(bm, 1.0)
@benchmark msd(bm, 1000.0)