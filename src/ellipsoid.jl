
"""
    Ellipsoid(v, A)

An ellipsoid defined by the equation `(x - v)ᵀΣ⁻¹(x - v) = 1`
which has volume V
"""
mutable struct Ellipsoid
    v::Vector{Float64}
    Σ::Matrix{Float64}
    Σ⁻¹::Matrix{Float64}
    V::Float64
end

function volume_hypersphere(d::Int)
    if d % 2 == 0
        return π ^ (d / 2) / factorial(d / 2)
    else
        return 2 ^ d * π ^ ((d - 1) / 2) * factorial((d - 1) / 2) / factorial(d)
    end
end

function volume_hyperellipsoid(Σ::Matrix{Float64})
    d = size(Σ, 1)
    return volume_hypersphere(d) * sqrt(det(Σ))
end

function volume_hyperellipsoid(E::Ellipsoid)
    d = length(E.v)
    return volume_hypersphere(d) * sqrt(det(E.Σ))
end


# find the bounding ellipsoid of points x where
function bounding_ellipsoid(x::Matrix{Float64}, enlarge=1.0)

    ndim, npoints = size(x)

    ctr = mean(x, 2)[:, 1]
    delta = x .- ctr
    cov = Base.unscaled_covzm(delta, 2)
    icov = inv(cov)

    # Calculate expansion factor necessary to bound each point.
    # This finds the maximum of (delta_i' * icov * delta_i)
    fmax = -Inf
    for k in 1:npoints
        f = 0.0
        for j=1:ndim
            for i=1:ndim
                f += icov[i, j] * delta[i, k] * delta[j, k]
            end
        end
        fmax = max(fmax, f)
    end

    fmax *= enlarge
    scale!(cov, fmax)
    scale!(icov, 1./fmax)
    vol = ellipsoid_volume(cov)

    return Ellipsoid(ctr, cov, icov, vol)
end
