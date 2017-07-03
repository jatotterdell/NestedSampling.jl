using Plots
gr()

function loglik(Θ)
    -((Θ[1] ^ 2 + Θ[2] - 11) ^ 2 + (Θ[1] + Θ[2] ^ 2 - 7) ^ 2)
end

function prior(Θ)
    return x
end

x = linspace(-5., 5., 100)
y = x
X = repmat(x', length(y), 1)
Y = repmat(y, 1, length(x))
Z = map((x,y) -> loglik([x,y]), X, Y)
contour(x, y, Z, fill = true, title = "Himmelbau's function")
surface(x, y, Z, zcolor = Z, title = "Himmelbau's function function")
