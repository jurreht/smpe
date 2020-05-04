module SMPE

using Cuba
using Distributed
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

# This allows us to comment out "using Distributed" during development
# to get serial execution no matter what.
if !@isdefined(Distributed)
    pmap(::AbstractWorkerPool, args) = map(args...)
    pmap(args...) = map(args...)
    macro sync(ex)
        return :( $(esc(ex)) )
    end
    macro distributed(ex)
        return :( $(esc(ex)) )
    end
end

export DynamicGame, SMPEOptions, compute_equilibrium, OptimizationConvergenceError

abstract type DynamicGame end

struct OptimizationConvergenceError <: Exception
    opt_results::Optim.OptimizationResults
end
Base.showerror(io::IO, e::OptimizationConvergenceError) = print(io, "optimization did not converge")

@Base.kwdef struct SMPEOptions
    eps_outer::Real = 1e-4
    eps_inner::Real = 1e-4
    eps_contraction::Real = 1e-4
    optim_method::Optim.FirstOrderOptimizer = LBFGS()
    optim_options::Optim.Options = Optim.Options()
    state_sim_eps::Real = 1e-2
    state_sim_min_sims::Integer = 100
    contraction_batch_size::Integer = 10000
    integration_rel_tol::Real = 1e-1
    integration_abs_tol::Real = 1e-3
end

DEFAULT_OPTIONS = SMPEOptions()

"""
Wrapper type for interpolated functions. Stores the game for which the function
is interpolated with the function. The purpose of this is to support games
which are analyzed on non-rectangular grids. Implemen transform_state()
and related methods, to transform from a rectangular to non-rectangular grid.
The domain of the wrapper type is the transformeed (non-rectangular) state and
it takes care of the chain rule when calculating the graient or hessian.
"""
struct TransformedInterpolation{S<:AbstractInterpolation{N, T, NT} where {N, T, NT}, G<:DynamicGame}
    interpolation::S
    game::G
end

(itp::TransformedInterpolation)(x...) = itp.interpolation(transform_state_back(itp.game, x)...)
Base.ndims(itp::TransformedInterpolation) = ndims(itp.interpolation)
Interpolations.gradient(itp::TransformedInterpolation, x...) = (
    transpose(transform_state_back_jac(itp.game, x)) *
    Interpolations.gradient(itp.interpolation, transform_state_back(itp.game, x)...)
)
function Interpolations.hessian(itp::TransformedInterpolation, x...)
    x_trans_back = transform_state_back(itp.game, x)
    grad_v = Interpolations.gradient(itp.interpolation, x_trans_back...)
    hess_v = Interpolations.hessian(itp.interpolation, x_trans_back...)
    jac_trans = transform_state_back_jac(itp.game, x)
    hess_trans = transform_state_back_hessian(itp.game, x)
    return (
        transpose(jac_trans) * hess_v * jac_trans +
        sum(hess_trans[i, :, :] .* grad_v[i] for i in 1:dim_rectangular_state(itp.game))
    )
end

mutable struct SolutionProgress{S <: AbstractArray{T, N} where {T <: Real, N}}
    calc_policy_functions::Vector{S}
    calc_value_functions::S
    prev_value_functions::S
    prev_optimal::Vector{Bool}
    attention::Matrix{Bool}
    converged::Vector{Bool}
    last_player::Int
    compute_time::Float64
end

function compute_equilibrium(
    game::DynamicGame,
    nodes::Union{AbstractRange, Vector{<:AbstractRange}};
    x0=nothing,
    options::SMPEOptions=DEFAULT_OPTIONS,
    return_interpolated=true,
    progress_cache::Union{IO, AbstractString, Nothing}=nothing
)
    if isa(nodes, AbstractRange)
        nodes = [nodes]
    end
    if size(nodes, 1) != dim_rectangular_state(game)
        throw(ArgumentError("State nodes must be of dimension dim_rectangular_state=$(dim_rectangular_state(game))"))
    end
    n_nodes = prod(length(s) for s in nodes)
    ALLNODES = fill(:, dim_rectangular_state(game))

    init_variables = true
    # progress = nothing
    if !isnothing(progress_cache)
        try
            progress = deserialize(progress_cache)
            @info("Restarting from progress cache")
            init_variables = false
        catch
            @info("Failed to load from progress cache, starting from scratch")
        end
    end
    if init_variables  # No progress cache given or progress cache is empty
        progress = SolutionProgress(
            create_init_actions(game, nodes, x0),
            initialize_value_functions(game, nodes),
            initialize_value_functions(game, nodes),  # It only matters that we get the shape right for now
            fill(isnothing(x0) ? false : true, num_players(game)),
            fill(true, (num_players(game), dim_state(game))),
            fill(false, num_players(game)),
            0,
            0.
        )
    end

    interp_policy_functions = map(
        i -> interpolate_policy_function(
            game,
            nodes,
            progress.calc_policy_functions[i]
        ),
        1:num_players(game)
    )
    interp_value_functions = map(
        i -> interpolate_value_function(
            game,
            nodes,
            progress.calc_value_functions[i, ALLNODES...]
        ),
        1:num_players(game)
    )

    # Pre-initialize value function if starting x were passed
    # Only do this if we started from a fresh progress cache, otherwise it
    # is wasted effort
    if !isnothing(x0) && progress.compute_time == 0
        @info "Initial contraction mapping"
        for i in 1:num_players(game)
            progress.calc_value_functions[i, ALLNODES...], interp_value_functions[i] = calculate_value_function(
                game,
                nodes,
                Iterators.product(nodes...),
                fill(true, dim_state(game)),
                i,
                interp_value_functions[i],
                interp_policy_functions,
                options
            )
        end
    end

    while !all(progress.converged)
        for player_ind = 1:num_players(game)
            if player_ind <= progress.last_player
                # If we restart from failure, start at the player where failure
                # occurred
                continue
            end

            progress.compute_time += @elapsed begin
                progress.calc_value_functions[player_ind, ALLNODES...], interp_value_functions[player_ind], progress.attention[player_ind, :] = innerloop_for_player!(
                    game,
                    nodes,
                    player_ind,
                    interp_value_functions[player_ind],
                    progress.calc_policy_functions,
                    interp_policy_functions,
                    progress.prev_optimal[player_ind],
                    options
                )

                if progress.prev_optimal[player_ind]  # At least one iteration done before => can check convergence
                    vf_norm = value_function_norm(
                        game,
                        progress.calc_value_functions[player_ind, ALLNODES...],
                        progress.prev_value_functions[player_ind, ALLNODES...]
                    )
                    @info "Outer loop step for player $(player_ind), step = $(vf_norm)"
                    if vf_norm < options.eps_outer
                        progress.converged[player_ind] = true
                    else
                        # The actions for one player have changed. As a
                        # result, we can no longer assume that the
                        # actions calculated for other players are optimal.
                        progress.converged = fill(false, num_players(game))
                    end
                end

                # After the first iteration, the values from the previous iteration
                # are probably good starting values for future iterations. Communicate
                # this to innerloop_for_player().
                progress.prev_optimal[player_ind] = true

                progress.prev_value_functions[player_ind, ALLNODES...] = progress.calc_value_functions[player_ind, ALLNODES...]

                progress.last_player = player_ind < num_players(game) ? player_ind : 0
            end


            if !isnothing(progress_cache)
                serialize(progress_cache, progress)
            end
        end
    end

    if return_interpolated
        return interp_policy_functions, interp_value_functions, progress.attention
    else
        return progress.calc_policy_functions, interp_value_functions, progress.attention
    end
end

function create_init_actions(game::DynamicGame, nodes, x0)
    if isnothing(x0)
        return map(
            i -> zeros((length(s) for s in nodes)..., num_actions(game, i)),
            1:num_players(game)
        )
    else
        return x0
        # TODO: Make this accept x0 in the shape of states
        throw("Not implemented atm")
        if isa(x0, Union{AbstractArray{<:Any, 2}, AbstractArray{<:Any, 3}})
        # Guarantee that x0 is Vector{Array{..., 2}}
            x0 = [x0[i, :, :] for i in 1:size(x0, 1)]
        end

        if length(x0) != num_players(game)
            throw(ArgumentError(x0, "Provide initial actions for every player."))
        end

        if any(size(x0[i], 2) != n_actions(game, i) for i in 1:num_players(game))
            throw(ArgumentError(
                x0,
                "Provide the correct number of initial actions for every player."
            ))
        end

        if any(size(x0[i], 1) != dim_state(game) for i in 1:num_players(game))
            throw(ArgumentError(x0, "Provide initial actions for every node."))
        end

        return x0
    end
end

initialize_value_functions(game::DynamicGame, nodes) = zeros(
    num_players(game),
    (length(node_base) for node_base in nodes)...
)

function innerloop_for_player!(
    game::DynamicGame,
    nodes,
    player_ind,
    interp_value_function,
    calc_policy_functions,
    interp_policy_functions,
    prev_optimal,
    options::SMPEOptions
)
    prev_value_func = nothing
    calc_value_func = nothing
    attention = nothing
    interp_value_func = nothing
    node_dims = (length(s) for s in nodes)

    @info "Inner loop for player $(player_ind)"

    while true
        @info "Calculating attention vector..."
        if attention_cost(game, player_ind) > 0
            default_state = collect(compute_default_state(game, player_ind))
            attention = calculate_attention(
                game,
                default_state,
                player_ind,
                interp_value_function,
                calc_policy_functions[1:end .!= player_ind],
                interp_policy_functions,
                options
            )
        else
            default_state = fill(NaN, dim_state(game))
            attention = fill(true, dim_state(game))
        end
        @debug "Attention vector = $(attention)"

        @info "Calculating policy function..."
        pool = CachingPool(workers())
        if all(attention)
            # The agent pays attention to the full state. This means that we
            # do not interpolation to predict the other agents' actions as the
            # relevant states will be precisely the elements in states.
            relevant_nodes = Iterators.product(nodes...)
            # Closed over voriables are only transmitted to workers once
            calc = let game = game, relevant_nodes = relevant_nodes, options=options, player_ind = player_ind, prev_optimal = prev_optimal, interp_value_function = interp_value_function, calc_policy_functions = calc_policy_functions
                pmap(
                    node_ind -> calculate_optimal_actions(
                        game,
                        transform_state(game, collect(relevant_nodes[node_ind])),
                        player_ind,
                        interp_value_function,
                        prev_optimal ? Slices(calc_policy_functions[player_ind], dim_rectangular_state(game) + 1)[node_ind] : nothing,
                        [Slices(pf, dim_rectangular_state(game) + 1)[node_ind] for pf in calc_policy_functions[1:end .!= player_ind]],
                        options
                    ),
                    pool,
                    # relevant_nodes supports fast linear indexing, while calc_policy_functions
                    # does not. Hence we get the indices from calc_policy_functions
                    # to get a CartesianIndex
                    eachindex(Slices(calc_policy_functions[1], dim_rectangular_state(game) + 1))
                )
            end
        else
            # Substitute default state for actual state if agent does not pay attention
            default_node = transform_state_back(game, default_state)
            relevant_nodes = Iterators.product((
                attention[i] ? nodes[i] : [default_node[i]]
                for i in 1:dim_rectangular_state(game)
            )...)
            calc = let game=game, player_ind=player_ind, prev_optimal=prev_optimal, interp_value_funtion=interp_value_function, options=options, interp_policy_functions=interp_policy_functions
                pmap(
                    state -> calculate_optimal_actions(
                        game,
                        state,
                        player_ind,
                        interp_value_function,
                        prev_optimal ? eval_policy_function(interp_policy_functions[player_ind], state) : nothing,
                        [eval_policy_function(pf, state) for pf in interp_policy_functions[1:end .!= player_ind]],
                        options
                    ),
                    pool,
                    map(node -> collect(transform_state(game, node)), relevant_nodes),
                )
            end
            # Repeat optimal policies across states that the agent does not pay attention to
            calc = repeat(calc; outer=[attention[i] ? 1 : length(nodes[i]) for i in 1:dim_rectangular_state(game)])
        end
        # We now have an Array{Array{..., 1}, n_dims}, but need a single
        # Array{..., n_actions + 1}. JuliennedArrays.Align() achieves this.
        calc_policy_functions[player_ind] = Align(calc, fill(False(), dim_rectangular_state(game))..., True())
        interp_policy_functions[player_ind] = interpolate_policy_function(
            game, nodes, calc_policy_functions[player_ind]
        )

        @info "Contraction mapping..."
        calc_value_func, interp_value_func = calculate_value_function(
            game,
            nodes,
            relevant_nodes,
            attention,
            player_ind,
            interp_value_function,
            interp_policy_functions,
            options
        )

        if !isnothing(prev_value_func)
            vf_norm = value_function_norm(game, calc_value_func, prev_value_func)
            @info "Step complete, norm = $(vf_norm)"
            if vf_norm < options.eps_inner
                @info "Inner loop converged"
                break
            end
        end

        prev_value_func = copy(calc_value_func)
        # After one iteration, can use prev optimum as starting point
        prev_optimal = true
    end

    return calc_value_func, interp_value_func, attention
end

function calculate_attention(
    game::DynamicGame,
    default_state,
    player_ind,
    interp_value_function,
    actions_others,
    interp_policy_funcs,
    options::SMPEOptions
)
    att_cost = attention_cost(game, player_ind)
    default_actions = calculate_optimal_actions(
        game,
        default_state,
        player_ind,
        interp_value_function,
        nothing,
        actions_others,
        options
    )
    @debug "Default state $(default_state), default actions $(default_actions)"
    n_actions_player = num_actions(game, player_ind)
    hess_value_actions = Matrix{Float64}(undef, n_actions_player, n_actions_player)

    # Calculate next_state here so that calculate_derivative_actions_state()
    # can dispath on its type
    next_state = compute_next_state(
        game,
        default_state,
        player_ind,
        default_actions,
        pad_actions(game, actions_others, player_ind)
    )

    deriv_actions_state = calculate_derivative_actions_state(
        game,
        default_state,
        next_state,
        player_ind,
        interp_value_function,
        default_actions,
        actions_others,
        options,
        hess_value_actions
    )
    state_var = simulate_state_variance(
        game,
        default_state,
        player_ind,
        default_actions,
        interp_policy_funcs,
        options
    )
    benefit_attention = -.5 * state_var .* diag(
        transpose(deriv_actions_state) *
        hess_value_actions *
        deriv_actions_state
    )
    @debug "Benefit attention = $(benefit_attention)"
    return benefit_attention .>= attention_cost(game, player_ind)
end

function calculate_derivative_actions_state(
    game::DynamicGame,
    state,
    ::AbstractVector{<:Real},
    player_ind,
    interp_value_function,
    actions_state,
    actions_others,
    options::SMPEOptions,
    hess_value_actions=nothing
)
    # By the implicit function theorem, da / ds = - (dg/da^{-1}) dg/ds,
    # where g() are the first order conditions for profit maximization. We need
    # to calculate a bunch of hessians to get there...
    x_to_args(x) = (
        game,
        x[1:dim_state(game)],
        player_ind,
        x[dim_state(game)+1:end],
        pad_actions(game, actions_others, player_ind)
    )

    x = vcat(state, actions_state)
    hess_static = ForwardDiff.hessian(x -> static_payoff(x_to_args(x)...), x)
    # We need both value and Jacobian for the state evolution, so calculate
    # calculate them in one go.
    state_evol_diff = DiffResults.JacobianResult(state, x)
    ForwardDiff.jacobian!(state_evol_diff, x -> compute_next_state(x_to_args(x)...), x)
    next_state = DiffResults.value(state_evol_diff)
    next_state_jac = DiffResults.gradient(state_evol_diff)
    # Hessians of vecor-valued functions are not supported, roll our own
    # (see http://www.juliadiff.org/ForwardDiff.jl/stable/user/advanced/#Hessian-of-a-vector-valued-function-1)
    next_state_hess = reshape(
        ForwardDiff.jacobian(
            x -> ForwardDiff.jacobian(x -> compute_next_state(x_to_args(x)...), x),
            x
        ),
        dim_state(game), length(x), length(x)
    )
    value_func_grad = Interpolations.gradient(interp_value_function, next_state...)
    value_func_hess = Interpolations.hessian(interp_value_function, next_state...)

    # Implicit function theorem incoming
    da = dim_state(game)+1:(dim_state(game) + num_actions(game, player_ind))
    ds = 1:dim_state(game)
    beta = discount_factor(game, player_ind)
    term1 = hess_static[da, da] + beta * (
        transpose(next_state_jac[:, da]) * value_func_hess * next_state_jac[:, da] +
        sum(next_state_hess[s, da, da] * value_func_grad[s] for s in 1:dim_state(game))
    )
    term2 = hess_static[da, ds] + beta * (
        transpose(next_state_jac[:, da]) * value_func_hess * next_state_jac[:, ds] +
        sum(next_state_hess[s, da, ds] * value_func_grad[s] for s in 1:dim_state(game))
    )

    if !isnothing(hess_value_actions)
        # Also return the Hessian of the agent's value with respect to its actions
        hess_value_actions[:, :] = term1
    end

    return -1 * term1 \ term2
end

function simulate_state_variance(
    game::DynamicGame,
    start_state::AbstractVector{T},
    player_ind,
    default_actions,
    interp_policy_funcs,
    options::SMPEOptions
) where {T}
    diff = Inf
    new_var = prev_var = fill(NaN, length(start_state))
    n_iters = 0
    states = fill(Vector{T}(), length(start_state))
    state = start_state
    while diff > options.state_sim_eps || n_iters < options.state_sim_min_sims
        actions = map(
            pfs_player -> eval_policy_function(pfs_player, state),
            interp_policy_funcs
        )
        actions[player_ind] = default_actions
        # TODO: Fix RNG?
        state = [
            isa(s, Sampleable) ? rand(s) : s
            for s in compute_next_state(game, state, player_ind, actions[player_ind], actions)
        ]
        for i in eachindex(states)
            push!(states[i], state[i])
        end
        new_var = [var(s) for s in states]
        diff = norm(new_var - prev_var)
        prev_var = new_var
        n_iters += 1
    end
    return new_var
end

function calculate_derivative_actions_state(
    game::DynamicGame,
    state,
    next_state::AbstractVector{<:Union{Real, ContinuousUnivariateDistribution}},
    player_ind,
    interp_value_function,
    actions_state,
    actions_others,
    options::SMPEOptions,
    hess_value_actions=nothing
)
    # By the implicit function theorem, da / ds = - (dg/da^{-1}) dg/ds,
    # where g() are the first order conditions for profit maximization. We need
    # to calculate a bunch of hessians to get there...
    n_actions = num_actions(game, player_ind)
    x_to_args(x) = (
        game,
        x[n_actions+1:end],
        player_ind,
        x[1:n_actions],
        pad_action(game, actions_others, player_ind)
    )

    x = vcat(actions_state, state)
    hess_static = ForwardDiff.hessian(x -> static_payoff(x_to_args(x)...), x)

    deterministic_states = [isa(s, Real) for s in next_state]
    deterministic_el_type = Union{(typeof(s) for s in next_state[deterministic_states])...}
    select_els = vcat(ones(Bool, n_actions), deterministic_states)


    next_state_jac = ForwardDiff.jacobian(
        x -> calculate_next_state_deterministic_part(
            deterministic_states,
            x_to_args(x)...
        ),
        x
    )[:, select_els]
    # Hessians of vecor-valued functions are not supported, roll our own
    # (see http://www.juliadiff.org/ForwardDiff.jl/stable/user/advanced/#Hessian-of-a-vector-valued-function-1)
    next_state_hess = reshape(
        ForwardDiff.jacobian(
            x -> ForwardDiff.jacobian(x -> calculate_next_state_deterministic_part(
                deterministic_states,
                x_to_args(x)...),
            x),
            x
        ),
        sum(deterministic_states), length(x), length(x)
    )[:, select_els, select_els]
    value_func_grad = value_function_gradient_for_state(
        game,
        interp_value_function,
        next_state,
        options)
    value_func_hess = value_function_hessian_for_state(
        game,
        interp_value_function,
        next_state,
        options)#[deterministic_states, deterministic_states]

    full_state = Vector{Union{deterministic_el_type, ForwardDiff.Dual}}(undef, length(state))
    full_state[deterministic_states] = state[deterministic_states]
    stochastic_states_hess = ForwardDiff.jacobian(
        state_stochastic -> begin
            full_state[.!deterministic_states] = state_stochastic
            next_state = compute_next_state(game, state, player_ind, actions_state, pad_actions(game, actions_others, player_ind))
            value_function_gradient_for_actions(game, state, next_state, player_ind, interp_value_function, actions_state, actions_others, options)
        end,
        state[.!deterministic_states]
    )

    # Implicit function theorem incoming
    da = 1:n_actions
    ds = (n_actions+1):size(hess_static, 1)
    n_deterministic = sum(deterministic_states)
    beta = discount_factor(game, player_ind)

    term1 = hess_static[da, da] + beta * (
        transpose(next_state_jac[:, da]) * value_func_hess * next_state_jac[:, da] +
        sum(next_state_hess[s, da, da] * value_func_grad[s] for s in 1:n_deterministic)
    )
    term2 = hess_static[da, ds]
    ds = (n_actions+1):size(next_state_jac, 2)
    term2[:, deterministic_states] += beta * (
        transpose(next_state_jac[:, da]) * value_func_hess * next_state_jac[:, ds] +
        sum(next_state_hess[s, da, ds] * value_func_grad[s] for s in 1:n_deterministic)
    )
    term2[:, .!deterministic_states] += beta * stochastic_states_hess

    if !isnothing(hess_value_actions)
        # Also return the Hessian of the agent's value with respect to its actions
        hess_value_actions[:, :] = term1
    end

    return -1 * term1 \ term2
end

eval_policy_function(policy_function, state) = map(
    pf -> pf(state...),
    policy_function
)

function calculate_optimal_actions(
    game::DynamicGame,
    state,
    player_ind,
    interp_value_function,
    actions_state,
    actions_others,
    options::SMPEOptions
)
    padded_actions = pad_actions(game, actions_others, player_ind)
    # The collect is necessary because of some JuliennedArrays magic in
    # innerloop_for_player!, which can cause a type mismatch in optimization.
    # collect(...) makes sure we have an Array{T, 1} here.
    x0 = collect(compute_optimization_x0(game, state, player_ind, interp_value_function, actions_state, padded_actions))
    bounds = compute_action_bounds(game, state, player_ind, interp_value_function, actions_state, padded_actions)
    return maximize_payoff(game, state, player_ind, interp_value_function, actions_others, x0, bounds, options)
end

function maximize_payoff(
    game::DynamicGame,
    state,
    player_ind,
    interp_value_function,
    actions_others,
    x0,
    bounds,
    options::SMPEOptions
)
    lb = [isnothing(bounds[i][1]) ? -Inf : bounds[i][1] for i in eachindex(bounds)]
    ub = [isnothing(bounds[i][2]) ? Inf : bounds[i][2] for i in eachindex(bounds)]

    if any(-Inf .< lb) || any(Inf .> ub)
        # Constrained optimization
        results = optimize(
            x -> calculate_negative_payoff(game, state, player_ind, interp_value_function, x, actions_others, options),
            (g, x) -> calculate_negative_payoff_gradient!(game, state, player_ind, interp_value_function, x, actions_others, g, options),
            lb,
            ub,
            x0,
            Fminbox(options.optim_method),
            options.optim_options
        )
    else
        # Unconstrained optimazation
        results = optimize(
            x -> calculate_negative_payoff(game, state, player_ind, interp_value_function, x, actions_others, options),
            (g, x) -> calculate_negative_payoff_gradient!(game, state, player_ind, interp_value_function, x, actions_others, g, options),
            x0,
            options.optim_method,
            options.optim_options
        )
    end
    if !Optim.converged(results) || !all([isfinite(x) for x in Optim.minimizer(results)])
        throw(OptimizationConvergenceError(results))
    end
    return Optim.minimizer(results)
end

function calculate_negative_payoff(
    game::DynamicGame,
    state,
    player_ind,
    interp_value_function,
    actions,
    actions_others,
    options::SMPEOptions
)
    padded_actions = pad_actions(game, actions_others, player_ind)
    payoff_now = static_payoff(game, state, player_ind, actions, padded_actions)
    delta = discount_factor(game, player_ind)
    if delta > 0
        next_state = compute_next_state(game, state, player_ind, actions, padded_actions)
        payoff = payoff_now + (
            delta *
            value_function_for_state(game, interp_value_function, next_state, options)
        )
    else
        payoff = payoff_now
    end
    return -1 * payoff
end

function calculate_negative_payoff_gradient!(
    game::DynamicGame,
    state,
    player_ind,
    interp_value_function,
    actions,
    actions_others,
    out,
    options::SMPEOptions
)
    padded_actions_others = pad_actions(game, actions_others, player_ind)
    payoff_now_grad = ForwardDiff.gradient(
        x -> static_payoff(
            game,
            state,
            player_ind,
            x,
            padded_actions_others
        ), actions
    )
    delta = discount_factor(game, player_ind)
    if delta > 0
        next_state = compute_next_state(game, state, player_ind, actions, padded_actions_others)
        vf_grad = value_function_gradient_for_actions(
            game,
            state,
            next_state,
            player_ind,
            interp_value_function,
            actions,
            actions_others,
            options
        )
        grad = payoff_now_grad + discount_factor(game, player_ind) * vf_grad
    else
        grad = payoff_now_grad
    end
    out[:] = -1 * grad
end

function pad_actions(game::DynamicGame, actions_others::AbstractVector{<:AbstractVector{T}}, player_ind)::Vector{<:AbstractVector{T}} where T
    return vcat(
        actions_others[1:player_ind-1],
        [zeros(T, num_actions(game, player))],
        actions_others[player_ind:end]
    )
end
pad_actions(game::DynamicGame, actions_others::AbstractVector{<:AbstractArray}, player_ind)::Vector{Vector{Float64}} = [zeros(num_actions(game, player_ind))]

interpolate_value_function(game::DynamicGame, states, calc_value_function) = interpolate_function(game, states, calc_value_function)

interpolate_policy_function(game::DynamicGame, states, calc_policy_function) = [
    interpolate_function(game, states, calc_policy_function[fill(:, length(states))..., i])
    for i in 1:size(calc_policy_function, length(states) + 1)
]

function interpolate_function(game::DynamicGame, states, vals)
    itp = interpolate(vals, BSpline(Cubic(Line(OnGrid()))))
    # We allow extrapolation to prevent errors when the due to floating point
    # rounding we are just outside the grid. It is not advised to try to
    # extrapolate much beyond that
    etp = extrapolate(Interpolations.scale(itp, states...), Interpolations.Flat())
    return TransformedInterpolation(etp, game)
end

function calculate_value_function(
    game::DynamicGame,
    nodes,
    relevant_nodes,
    attention,
    player_ind,
    interp_value_function,
    interp_policy_functions,
    options::SMPEOptions
)
    calc_value_function = nothing
    prev_value_function = nothing

    # Cache static payoffs and state transitions
    pool = CachingPool(workers())
    relevant_states = map(node -> transform_state(game, node), relevant_nodes)
    actions_grid = map(
        pf -> pmap(
            state -> eval_policy_function(pf, state),
            pool,
            relevant_states
        ),
        interp_policy_functions
    )
    static_payoff_grid = let game=game, player_ind=player_ind
        pmap(
            zipped -> static_payoff(game, zipped[1], player_ind, zipped[2:end][player_ind], zipped[2:end]),
            pool,
            zip(relevant_states, actions_grid...)
        )
    end
    next_state_grid = let game=game, player_ind=player_ind
        pmap(
            zipped -> compute_next_state(game, zipped[1], player_ind, zipped[2:end][player_ind], zipped[2:end]),
            pool,
            zip(relevant_states, actions_grid...)
        )
    end
    # # Allow the GC to take care of this
    relevant_states = nothing
    actions_grid = nothing

    repeat_calc_vf = [
        indim == 1 ? outdim : 1
        for (indim, outdim)
        in zip(size(relevant_nodes), (length(s) for s in nodes))
    ]

    while true
        calc_value_function = let game=game, player_ind=player_ind, options=options, interp_value_function=interp_value_function
            pmap(
                zipped -> map(
                    z -> calculate_value(
                        game,
                        z[2],
                        player_ind,
                        interp_value_function,
                        z[1],
                        options
                    ),
                    zipped
                ),
                pool,
                Iterators.partition(
                    zip(static_payoff_grid, next_state_grid),
                    options.contraction_batch_size
                )
            )
        end
        calc_value_function = reshape(
            vcat(calc_value_function...),
            size(relevant_nodes)
        )
        calc_value_function = repeat(calc_value_function, repeat_calc_vf...)

        if !isnothing(prev_value_function)
            diff = value_function_norm(game, calc_value_function, prev_value_function)
            @debug "Contraction mapping step = $(diff)"
            if diff < options.eps_contraction
                break
            end
        end
        interp_value_function = interpolate_value_function(game, nodes, calc_value_function)
        prev_value_function = copy(calc_value_function)
    end
    return calc_value_function, interp_value_function
end

perceived_state(game::DynamicGame, state, attention, default_state) = map(
    (s, att, ds) -> att ? s : ds,
    state,
    attention,
    default_state
)

function calculate_value(
    game::DynamicGame,
    next_state,
    player_ind,
    interp_value_function,
    static_payoff,
    options::SMPEOptions
)
    ret = static_payoff
    delta = discount_factor(game, player_ind)
    if delta > 0
        ret += delta * value_function_for_state(
            game,
            interp_value_function,
            next_state,
            options
        )
    end
    return ret
end

function value_function_for_state(
    game::DynamicGame,
    interp_value_function::TransformedInterpolation{<:AbstractInterpolation{T, N, NT}, <:DynamicGame},
    state::AbstractVector{<:Real},
    options::SMPEOptions
)::T where {T<:Real, N, NT}
    interp_value_function(state...)
end

function value_function_for_state(
    game::DynamicGame,
    interp_value_function::TransformedInterpolation{<:AbstractInterpolation{T, N, NT}, <:DynamicGame},
    state::AbstractVector{<:Union{<:Real, ContinuousUnivariateDistribution}},
    options::SMPEOptions
)::T where {T<:Real, N, NT}
    deterministic_states = map(s -> isa(s, T), state)
    node = transform_state_back(game, state)
    deterministic_nodes = map(s -> isa(s, T), node)
    return integrate_fn_over_stochastic_states(
        game,
        # We need an Array here since that is what is needed for numerical integration
        x -> [value_function_for_state(game, interp_value_function, x, options)],
        1,
        interpolation_bounds(interp_value_function),
        node,
        deterministic_nodes,
        options
    )[1]
end

function value_function_gradient_for_state(
    game::DynamicGame,
    interp_value_function::TransformedInterpolation{<:AbstractInterpolation{T, N, NT}, <:DynamicGame},
    state::AbstractVector{<:Real},
    options::SMPEOptions
)::Vector{T} where {T<:Real, N, NT}
    Interpolations.gradient(interp_value_function, state...)
end

function value_function_gradient_for_state(
    game::DynamicGame,
    interp_value_function::TransformedInterpolation{<:AbstractInterpolation{T, N, NT}, <:DynamicGame},
    state::AbstractVector{<:Union{Real, ContinuousUnivariateDistribution}},
    options::SMPEOptions
)::Vector{T} where {T<:Real, N, NT}
    deterministic_states = map(s -> isa(s, T), state)
    node = transform_state_back(game, state)
    deterministic_nodes = map(s -> isa(s, T), node)
    return integrate_fn_over_stochastic_states(
        game,
        x -> value_function_gradient_for_state(game, interp_value_function, x, options)[deterministic_states],
        sum(deterministic_states),
        interpolation_bounds(interp_value_function),
        node,
        deterministic_nodes,
        options
    )
end

function value_function_gradient_for_actions(
    game::DynamicGame,
    state::AbstractVector{T},
    next_state::AbstractVector{<:Real},
    player_ind,
    interp_value_function::TransformedInterpolation{<:AbstractInterpolation{T, N, NT}, <:DynamicGame},
    actions,
    actions_others,
    options::SMPEOptions
)::Vector{T} where {T<:Real, N, NT}
    next_state_jac = ForwardDiff.jacobian(
        x -> compute_next_state(
            game,
            state,
            player_ind,
            x,
            pad_actions(game, actions_others, player_ind)
        ),
        actions
    )
    next_vf_grad = value_function_gradient_for_state(game, interp_value_function, next_state, options)
    Interpolations.gradient(interp_value_function, state...)
    return transpose(next_state_jac) * next_vf_grad
end

function value_function_gradient_for_actions(
    game::DynamicGame,
    state::AbstractVector{T},
    next_state::AbstractVector{<:Union{<:Real, ContinuousUnivariateDistribution}},
    player_ind,
    interp_value_function::TransformedInterpolation{<:AbstractInterpolation{T, N, NT}, <:DynamicGame},
    actions,
    actions_others,
    options::SMPEOptions
)::Vector{T} where {T<:Real, N, NT}
    next_node = transform_state_back(game, next_state)
    # Only differentiate wrt the non-stochastic parts of the state/node
    # (i.e. it is assumed that the stochastic state transitions are exogenous)
    deterministic_states = map(s -> isa(s, T), next_state)
    deterministic_nodes = map(s -> isa(s, T), next_node)
    # We can hence take the Jacobian of the state evolution wrt to
    # the deterministic states only
    next_state_jac = ForwardDiff.jacobian(
        x -> calculate_next_state_deterministic_part(
            deterministic_states,
            game,
            state,
            player_ind,
            x,
            pad_actions(game, actions_others, player_ind)
        ),
        actions
    )

    # Integrate out the random variables in the gradient of the value function
    # Integration needs [0, 1] box => use a change of variable
    next_vf_grad = integrate_fn_over_stochastic_states(
        game,
        x -> value_function_gradient_for_state(game, interp_value_function, x, options)[deterministic_states],
        sum(deterministic_states),
        interpolation_bounds(interp_value_function),
        next_node,
        deterministic_nodes,
        options
    )
    return transpose(next_state_jac) * next_vf_grad
end

function calculate_next_state_deterministic_part(
    deterministic_states, args...
)
    ret = compute_next_state(args...)[deterministic_states]
    # Elements of ret will be Union{<:Real, <:ContinuousUnivariateDistribution}
    # But the deterministic part is only <:Real. Hence convert types here.
    # If we don't do this, we get type conversion errors in ForwardDiff.
    eltype = Union{(typeof(x) for x in ret)...}
    return convert(Vector{eltype}, ret)
end

function integrate_fn_over_stochastic_states(
    game::DynamicGame,
    fn,
    dim_fn,
    domain_fn,
    node,
    deterministic_nodes,
    options::SMPEOptions
)
    stochastic_nodes = node[.!deterministic_nodes]
    change_var = map(
        x -> make_scaler(x...),
        zip(stochastic_nodes, domain_fn[.!deterministic_nodes])
    )
    integrand_node = [
        is_deterministic ? x : NaN
        for (is_deterministic, x) in
        zip(deterministic_nodes, node)
    ]
    integral = vegas(
        (x, f) -> begin
            integrand_node[.!deterministic_nodes] = [
                cv[1](y) for (cv, y) in zip(change_var, x)
            ]
            integrand_state = transform_state(game, integrand_node)
            transform_jac = transform_state_jac(game, integrand_node)
            cv_jac = prod(cv[2](y) for (cv, y) in zip(change_var, x))
            weight = prod(pdf(s, y) for (s, y) in zip(stochastic_nodes, x))
            f[:] = fn(integrand_state) * cv_jac * weight * abs(det(transform_jac))
        end,
        length(stochastic_nodes),
        dim_fn;
        rtol=options.integration_rel_tol,
        atol=options.integration_abs_tol
    )
    if integral.fail > 0
        throw("Integration of future states failed")
    end
    return integral.integral
end

"""
For a given ContinuousUnivariateDistribution, return a change of variables
with support [0, 1]. The first element of the return is the change of variables,
the second the Jacobian.
"""
function make_scaler(dist::ContinuousUnivariateDistribution, domain)
    # Make sure we do not integrate outside of the domain of the function.
    # This is important because value functions are interpolated on a grid.
    # We do not have a good idea of the value outside that grid, so we simply
    # do not integrate there. Of course, this only works if the probability that
    # the state variable falls outside the domain is small.
    lower_support = max(minimum(dist), domain[1])
    upper_support = min(maximum(dist), domain[2])
    if isfinite(lower_support) && isfinite(upper_support)
        return (
            x -> lower_support + (upper_support - lower_support) * x,
            x -> upper_support - lower_support
        )
    elseif isfinite(lower_support) && !isfinite(upper_support)
        return (
            x -> lower_support + x / (1 - x),
            x -> 1 / (1 - x)^2
        )
    else
        throw("This type of support is not implemented yet")
    end
end

function value_function_hessian_for_state(
    game::DynamicGame,
    interp_value_function::TransformedInterpolation{<:AbstractInterpolation{T, N, NT}, <:DynamicGame},
    states::AbstractVector{T},
    options::SMPEOptions
)::Matrix{T} where {T<:Real, N, NT}
    # There is no analytic hessian available for extrapolated functions
    # There is also a bug in Interpolations.jl for computing the Hssian when
    # ndim >= 4. See https://github.com/JuliaMath/Interpolations.jl/issues/364
    # So we just use ForwardDiff
    return ForwardDiff.hessian(x -> interp_value_function(x...), states)
end

function value_function_hessian_for_state(
    game::DynamicGame,
    interp_value_function::TransformedInterpolation{<:AbstractInterpolation{T, N, NT}, <:DynamicGame},
    state::AbstractVector{<:Union{Real, ContinuousUnivariateDistribution}},
    options::SMPEOptions
)::Matrix{T} where {T<:Real, N, NT}
    node = transform_state_back(game, state)
    deterministic_states = map(s -> isa(s, T), state)
    deterministic_nodes = map(s -> isa(s, T), node)
    n_deterministic = sum(deterministic_states)
    hess_vectorized = integrate_fn_over_stochastic_states(
        game,
        # Integration only possible over vectors. Hence vectorize and reshape below
        x -> vec(value_function_hessian_for_state(game, interp_value_function, x, options)[deterministic_states, deterministic_states]),
        n_deterministic^2,
        interpolation_bounds(interp_value_function),
        node,
        deterministic_nodes,
        options
    )
    return reshape(hess_vectorized, n_deterministic, n_deterministic)
end

function interpolation_bounds(interp::TransformedInterpolation)
    itp = interp.interpolation
    # Extrapolated functions have no bounds, give the bounds of the
    # underlying interpolated function
    vf_bounds = (isa(itp, AbstractExtrapolation)
        ? bounds(parent(itp))
        : bounds(itp)
    )
    return collect(zip(
        map(minimum, vf_bounds),
        map(maximum, vf_bounds)))
end

compute_optimization_x0(
    game::DynamicGame, state, player_ind, interp_value_function, actions_player, actions
) = isnothing(actions_player) ? zeros(num_actions(game, player_ind)) : actions_player

compute_action_bounds(
    game::DynamicGame, state, player_ind, interp_value_function, actions_player, actions
) = fill((nothing, nothing), num_actions(game, player_ind))

# Convergence criterion as in Doraszelski & Pakes (2007)
value_function_norm(game::DynamicGame, new_value, old_value) = max(
    map(abs, (new_value .- old_value) ./ (1 .+ new_value))...
)

# getindex() is not implemented for either Enumerate or ProductIterator.
# But @distributed() needs it to loop over the states. So we have it...
function Base.getindex(
    it::Base.Iterators.ProductIterator{<:Any},
    ind::Integer
)
    itrs = it.iterators
    ret = []
    prodlengths = 1
    for i in 1:length(itrs)
        j = 1 + mod(ceil(Integer, ind / prodlengths) - 1, length(itrs[i]))
        push!(ret, itrs[i][j])
        prodlengths *= length(itrs[i])
    end
    return tuple(ret...)
end
Base.getindex(
    it::Base.Iterators.ProductIterator{<:Any},
    ind::CartesianIndex
) = tuple((it.iterators[i][ind.I[i]] for i in 1:length(it.iterators))...)

Base.getindex(
    it::Base.Iterators.ProductIterator{<:Any},
    ind::UnitRange
) = [getindex(it, i) for i in ind]
Base.keys(it::Base.Iterators.ProductIterator{<:Any}) = 1:length(it)

Base.getindex(it::Base.Iterators.Enumerate{<:Any}, ind::Integer) = (ind, getindex(it.itr, ind))
Base.getindex(it::Base.Iterators.Enumerate{<:Any}, ind::UnitRange) = [
    (i, getindex(it.itr, i)) for i in ind
]
Base.keys(it::Base.Iterators.Enumerate{<:Base.Iterators.ProductIterator{<:Any}}) = 1:length(it)



#
# The methods below must be implemented by the DynamicGame concrete type
#
num_players(g::DynamicGame) = throw("Concrete subtypes of DynamicGame must implement num_players()")
dim_state(g::DynamicGame) = throw("Concrete subtypes of DynamicGame must implement dim_state()")
num_actions(g::DynamicGame, player_ind) = throw("Concrete subtypes of DynamicGame must implement num_actions()")
discount_factor(g::DynamicGame, player_ind) = throw("Concrete subtypes of DynamicGame must implement discount_factor()")
static_payoff(g::DynamicGame, state, player_ind, actions_player, actions_others) = throw("Concrete subtypes of DynamicGame must implement static_payoff()")
compute_next_state(g::DynamicGame, state, player_ind, actions, actions_others) = throw("Concrete subtypes of DynamicGame must implement compute_next_state()")
attention_cost(g::DynamicGame, player_ind) = throw("Concrete subtypes of DynamicGame must implement attention_cost()")
compute_default_state(g::DynamicGame, player_ind) = throw("Concrete subtypes of DynamicGame must implement compute_default_state() when attention costs are positive")

# Optional overrides
dim_rectangular_state(g::DynamicGame) = dim_state(g)
transform_state(g::DynamicGame, state) = state
# Note this Jacobian *should* be square. So if the transformation node -> state
# changes the number of dimensions, do not include the derivatives wrt linearly
# dependent states.
transform_state_jac(g::DynamicGame, state) = I
transform_state_back(g::DynamicGame, state) = state
transform_state_back_jac(g::DynamicGame, state) = I
transform_state_back_hessian(g::DynamicGame, state) = fill(I, dim_rectangular_state(g))

# Helper fuctions
# Transformation rectangular grid <> simplex
# Based on https://math.stackexchange.com/a/1945720/404905
function rectangular_to_simplex(state)
    state = [state[i] * prod(1 .- state[i+1:end]) for i in eachindex(state)]
    return vcat(state, [1 - sum(state)])
end
function rectangular_to_simplex_jac(state)
    ret = zeros(length(state), length(state))
    for i in eachindex(state)
        ret[i, i:end] .= prod(1 .- state[i+1:end])
        ret[i, (i+1):end] ./= -1 .* (1 .- state[i+1:end])
    end
    return ret
end
function simplex_to_rectangular(state)
    state = state[1:end-1]
    return [state[i] / (1 - sum_state_others(state, i)) for i in eachindex(state)]
end
function simplex_to_rectangular_jac(state)
    state = state[1:end-1]
    ret = zeros(length(state), length(state) + 1)
    for i in eachindex(state)
        ret[i, i] = 1 / (1 - sum_state_others(state, i))
        ret[i, i+1:(end-1)] .= state[i] / (1 - sum_state_others(state, i))^2
    end
    return ret
end
function simplex_to_rectangular_hess(state)
    state = state[1:end-1]
    ret = zeros(length(state), length(state) + 1, length(state) + 1)
    for i in eachindex(state)
        ret[i, i, i+1:(end-1)] .= 1 / (1 - sum_state_others(state, i))^2
        ret[i, i+1:(end-1), i] .= 1 / (1 - sum_state_others(state, i))^2
        ret[i, i+1:(end-1), i+1:(end-1)] .= 2 * state[i] / (1 - sum_state_others(state, i))^3
    end
    return ret
end
# We need this function because the sum of an empty list is not defined in Julia
sum_state_others(state, i) = i < length(state) ? sum(state[i+1:end]) : 0.

end
