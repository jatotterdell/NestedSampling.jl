using Base.Test
using TimeIt
import NestedSampling: volume_hypersphere, sample

###############################
## Test volume for an N-ball ##
###############################
@test volume_hypersphere(1) == 2.0
@test volume_hypersphere(2) ≈ π
@test volume_hypersphere(3) ≈ 4/3 * π
@test volume_hypersphere(4) ≈ π^2 / 2

# -----------------------------------------------------------------------------
# Simple likelihood function

    # gaussians centered at (1, 1) and (-1, -1) with a width of 0.1.
const mu1 = [1., 1.]
const mu2 = [-1., -1.]
const sigma = 0.1
const invvar = eye(2) / sigma^2

function logl(x)
    dx1 = x .- mu1
    dx2 = x .- mu2
    return logaddexp((dx1' * invvar * dx1)[1] / 2.0,
                     (dx2' * invvar * dx2)[1] / 2.0)
end



# Use a flat prior, over [-5, 5] in both dimensions
prior(x) = 10.0 .* x .- 5.0

srand(0)
res = sample(logl, prior, 2; npoints=100)
@printf "evidence = %6.3f +/- %6.3f\n" res["logz"] res["logzerr"]

#(Approximate) analytic evidence for two identical Gaussian blobs,
# over a uniform prior [-5:5][-5:5] with density 1/100 in this domain:
analytic_logz = log(2.0 * 2.0*pi*sigma*sigma / 100.)
@printf "analytic = %6.3f\n" analytic_logz
