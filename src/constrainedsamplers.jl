abstract type ConstrainedSampler end
abstract type LocalStepSampler <: ConstrainedSampler end
abstract type RegionSampler <: ConstrainedSampler end

struct SymmetricSampler <: LocalStepSampler
    propose_point::Function
end

struct EllipsoidSampler <: RegionSampler

end


mutable struct GaussianRW <: LocalStepSampler
    constraint::Function
    scale::Float64
    proposal::MvNormal
    propose_point::Function
    sampled_point::Function
end
function GaussianRW(loglikelihood::Function, livepoints::Array, threshold)
    d = size(livepoints, 1)
    scale = 2.56 ^ 2 / d
    Σ = cov(livepoints')
    constraint = (x) -> loglikelihood(x) > threshold
    proposal = MvNormal(livepoints[:, rand(indices(livepoints, 2))], Σ)

    function propose_point(current, scale)
        current + scale * rand(proposal)
    end

    function sampled_point(current)
        oldθ = current
        accept = 0
        reject = 0
        ncall  = 0
        while accept < 20
            newθ = propose_point(oldθ, scale)

            if accept > reject
                scale *= exp(1 / accept)
            elseif accept < reject
                scale *= exp(-1 / reject)ind
            end
        end
        return newθ, newl, ncall
    end

    GaussianRW(constraint, scale, proposal, propose_point, sampled_point)
end

function draw_constrained!(loglik::Function, logprior::Function, livepoints, L, i)

    d, N = size(livepoints)
    Σ = cov(livepoints[:, setdiff(1:end, i)]')
    k = rand(setdiff(1:N, i))
    livepoints[:, i] = livepoints[:, k]
    l = L[i]
    L[i] = L[k]
    scale = 2.38^2 / d

    accept = 0
    reject = 0
    while(accept < 20)

        # Generation of new samples...
        newpoint = rand(MvNormal(livepoints[:, i], scale * Σ))
        a_prob = min(0, logprior(newpoint) - logprior(livepoints[:, i]))
        if log(rand()) <= a_prob && loglik(newpoint) > l

            livepoints[:, i] = newpoint
            L[i] = loglik(newpoint)
            accept += 1

        else
            reject += 1
        end

        if accept > reject
            scale *= exp(1 / accept)
        elseif accept < reject
            scale *= exp(-1 / reject)
        end
    end
    accept + reject
end
