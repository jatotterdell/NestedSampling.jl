include("../src/NestedSampling.jl")
using Plots
gr()

# Define the posterior density to be sampled:
tmax = 10.0 * π
constant = log(1.0 / tmax ^ 2)

function loglik(t)
    return (2.0 + cos(t[1] / 2.0) * cos(t[2] / 2.0)) ^ 5.0
end

function logprior(θ)
    return logpdf(Uniform(0, tmax), θ[1]) + logpdf(Uniform(0, tmax), θ[2])
end

function randprior(n::Int)
    vcat(rand(Uniform(0.0, tmax), n)', rand(Uniform(0.0, tmax), n)')
end

hcubature((x) -> loglik(x) + logprior(x), [0.0, 0.0], [tmax, tmax])

out = NestedSampling.nested_sampling(loglik, logprior, randprior, 2, draw_constrained!, nlive=500, maxiter = 50000)

x = linspace(0., tmax, 100)
y = x
X = repmat(x', length(y), 1)
Y = repmat(y, 1, length(x))
Z = map((x,y) -> loglik([x,y]), X, Y)

surface(x, y, Z, zcolor = Z, title = "Egg box function", legend=false)
scatter(out[:samples][1, :], out[:samples][2, :], out[:logl], zcolor = out[:logl], markersize = 2, markerstrokewidth=0.2, legend=false)
