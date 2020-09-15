module SMPE

using Cuba
using Distributions
using ForwardDiff
using Logging
using Interpolations
using JuliennedArrays
using LinearAlgebra
using Optim
using PyCall
using Random
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

const State = Vector{Float64}
const Actions = Vector{Vector{Float64}}
const EquilibriumPath = Vector{Tuple{State, Actions}}

const moment_inequality = PyNULL()

function __init__()
    pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__)
    copy!(moment_inequality, pyimport("moment_inequality"))
end

include("game.jl")
include("common.jl")
include("compute.jl")
include("estimation.jl")
include("simulate.jl")

export DynamicGame, SMPEOptions, compute_equilibrium, OptimizationConvergenceError, simulate_equilibrium_path, estimate_attention_cost

end
