using Plots
gr()

function loglik(Θ)
    -sum((1 - Θ[1:(end-1)]) .^ 2 + 100 * (Θ[end] - Θ[1:(end-1)] .^ 2) .^ 2)
end

x = linspace(-3., 4., 100)
y = linspace(0., 10., 100)
X = repmat(x', length(y), 1)
Y = repmat(y, 1, length(x))
Z = map((x,y) -> loglik([x,y]), X, Y)
contour(x, y, Z, fill = true, title = "Rosenbrock function")
surface(x, y, Z, zcolor = Z, title = "Rosenbrock function")
