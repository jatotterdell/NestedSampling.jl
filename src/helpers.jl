"""
    randnsphere(ndim)

Returns a point on the ndim unit hypersphere
"""
function randnsphere(ndim::Int)
    z = randn(ndim)
    return z ./ norm(z, 2)
end

"""
    randnsphere(ndim, n)

Returns a sample of size n on the ndim unit hypersphere
"""
function randnsphere(ndim::Int, n::Int)
    out = Array{Float64}(ndim, n)
    for i in 1:n
        out[:, i] = randnsphere(ndim)
    end
    return out
end

"""
    randnball(ndim)

Returns a point in the ndim unit hyperball
"""
function randnball(ndim::Int)
    r = rand() ^ (1 / ndim)
    return r * randnsphere(ndim)
end

"""
    randnball(ndim, n)

Returns a sample of size n in the ndim unit hyperball
"""
function randnball(ndim::Int, n::Int)
    out = Array{Float64}(ndim, n)
    for i in 1:n
        out[:, i] = randnball(ndim)
    end
    return out
end

"""
    randnellipsoid(ndim, Σ, r)
    randnellipsoid(ndim, B, r)

Hyperellipsoid defined by the equation `xᵀΣx = r²`.

Returns a point in the hyperellipsoid with axes determined by
the matrix Σ or the lower triangular matrix B.
"""
function randnellipsoid(ndim::Int, Σ::Matrix, r::Int = 1)
    if !isposdef(Σ)
        error("Σ must be positive definite.")
    end
    B = cholfact(Σ)[:U]
    z = randnball(ndim)
    return B \ (r * z)
end

function randnellipsoid(ndim::Int, Σ::Matrix, r::Int = 1)
    if !isposdef(Σ)
        error("Σ must be positive definite.")
    end
    v = eigfact(Σ)[:vectors]
    z = randnball(ndim)
    return v * z
end

function randnellipsoid(ndim::Int, B::UpperTriangular, r::Int = 1)
    z = randnball(ndim)
    return B \ (r * z)
end

function randnellipsoid(ndim::Int, n::Int, Σ::Matrix, r::Int = 1)
    if !isposdef(Σ)
        error("Σ must be positive definite.")
    end
    B = cholfact(Σ)[:U]
    out = Array{Float64, 2}(ndim, n)
    for i in 1:n
        out[:, i] = randnellipsoid(ndim, B, r)
    end
    return out
end

function randnellipsoid(ndim::Int, n::Int, Σ::Matrix, r::Int = 1)
    if !isposdef(Σ)
        error("Σ must be positive definite.")
    end
    out = Array{Float64, 2}(ndim, n)
    f = eigfact(Σ)
    v, w = f[:vectors], f[:values]
    for j=1:ndim
        tmp = sqrt(w[j])
        for i=1:ndim
            v[i, j] *= tmp
        end
    end
    for i in 1:n
        out[:, i] = v * randnball(ndim)
    end
    return out
end
