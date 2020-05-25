module SMPE

using Cuba
using Distributions
using ForwardDiff
using Logging
using Interpolations
using JuliennedArrays
using LinearAlgebra
using Optim
using Serialization
using Statistics

# This allows to comment away "using Logging" above and have working code
# with some messages. This is useful during development, since the Juno editor
# has some issues with finding the right line when a macro causes a multi-line
# expansion.
if !@isdefined(Logging)
    macro info(ex)
        return :( println($(esc(ex))) )
    end

    macro debug(ex)
        return :( println($(esc(ex))) )
    end
end

include("game.jl")
include("common.jl")
include("compute.jl")

export DynamicGame, SMPEOptions, compute_equilibrium, OptimizationConvergenceError

end
