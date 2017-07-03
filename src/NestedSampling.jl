module NestedSampling

    #############
    ## IMPORTS ##
    #############

    import StatsBase: weights
    import StatsFuns: logsumexp
    import Distributions: MvNormal

    ###########
    ## TYPES ##
    ###########


    ##############
    ## INCLUDES ##
    ##############

    include("helpers.jl")
    include("ellipsoid.jl")
    include("constrainedsamplers.jl")

    #############
    ## EXPORTS ##
    #############


    function nested_sampling(loglikelihood::Function, logprior::Function, randprior::Function,
                             ndim::Int, draw_constrained!::Function; nlive::Int=100, maxiter::Int=10000)

        # Choose initial points and calculate likelihoods
        liveθ       = randprior(nlive)                          # position of live pts in prior space
        liveloglike = mapslices(loglikelihood, liveθ, 1)[1, :]  # log(likelihood) at each point
        ncall = nlive                                           # number of likelihood calls

        # Initialize values for nested sampling loop.
        deadθ        = Array{Float64, 2}(ndim, maxiter + nlive)
        deadloglike  = Float64[]
        deadlogprior = Float64[]
        deadlogwt    = Float64[]
        ℓ            = 0.   # ln(Likelihood constraint)
        H            = 0.   # Information, initially 0.
        logz         = -1e300 # log(evidence Z), initially 0

        # ln(width in prior mass), outermost width is 1 - e^(-1/n)
        logwidth = log(1. - exp(-1. / nlive))

        # Nested sampling loop.
        ndecl = 0
        logwt_old = -Inf
        niter = 0
        while niter < maxiter
            niter += 1

            # find lowest logl in active points
            ℓ, minidx = findmin(liveloglike)
            logwt = logwidth + ℓ

            # update evidence and information

            logz_new = logsumexp(logz, logwt)
            H = (exp(logwt - logz_new) * ℓ +
                 exp(logz - logz_new) * (H + logz) - logz_new)
            logz = logz_new

            # Add worst object to samples.
            deadθ[:, niter] =  liveθ[:, minidx]
            push!(deadlogwt, logwt)
            push!(deadlogprior, logwidth)
            push!(deadloglike, ℓ)

            expected_vol = exp(-niter / nlive)

            # Sample a new constrained point
            ncall += draw_constrained!(loglikelihood, logprior, liveθ, liveloglike, minidx)

            # Shrink interval
            logwidth -= 1. / nlive

            # stop when logwt has been declining for more than 2*npoints
            # or niter/6 consecutive iterations.
            ndecl = (logwt < logwt_old) ? ndecl + 1 : 0
            if (ndecl > 2 * nlive) && (ndecl > niter / 6)
                break
            end
            logwt_old = logwt
        end

        # Add remaining objects.
        # After N samples have been taken out, the remaining width is e^(-N/nobj)
        # The remaining width for each object is e^(-N/nobj) / nobj
        # The log of this for each object is:
        # log(e^(-N/nobj) / nobj) = -N/nobj - log(nobj)
        # nsamples = length(liveθ) ÷ ndim
        # logwidth = -nsamples / nlive - log(nlive)
        # for i in 1:nlive
        #     logwt = logwidth + liveloglike[i]
        #     logz_new = logsumexp(logz, logwt)
        #     H = (exp(logwt - logz_new) * liveloglike[i] +
        #          exp(logz - logz_new) * (H + logz) - logz_new)
        #     logz = logz_new
        #
        #     deadθ[:, (niter + i)] =  liveθ[:, i]
        #     push!(deadlogwt, logwt)
        #     push!(deadloglike, liveloglike[i])
        #     push!(deadlogprior, logwidth)
        # end
        #
        # nsamples += nlive

        return Dict(:niter => niter,
                    :ncall => ncall,
                    :logz => logz,
                    :logzerr => sqrt(H/nlive),
                    :loglmax => maximum(liveloglike),
                    :H => H,
                    :samples => deadθ[:, 1:(niter)],
                    :weights => weights(exp.(deadlogwt .- logz)),
                    :logprior => deadlogprior,
                    :logl => deadloglike)
    end

end
