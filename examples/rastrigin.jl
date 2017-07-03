using Plots
gr()

function loglik(Θ)
    -(10 * 2 + sum(Θ .^ 2 - 10 * cos.(2 * π * Θ)))
end

function logprior(θ)
    return loglikelihood(Uniform(-5.12, 5.12), θ)
end

function randprior(n::Int)
    vcat(rand(Uniform(-5.12, 5.12), n)', rand(Uniform(-5.12, 5.12), n)')
end

out = NestedSampling.nested_sampling(loglik, logprior, randprior, 2, draw_constrained!, nlive=1000, maxiter = 200000)


x = linspace(-5.12, 5.12, 100)
y = x
X = repmat(x', length(y), 1)
Y = repmat(y, 1, length(x))
Z = map((x,y) -> loglik([x,y]), X, Y)
contour(x, y, Z, fill = true, title = "Rastrigin function")
surface(x, y, Z, zcolor = Z, title = "Rastrigin function")
scatter!(out[:samples][1, :], out[:samples][2, :], out[:logl],
         zcolor = out[:logl], markersize = 2, markerstrokewidth = 0.25, legend = false)
