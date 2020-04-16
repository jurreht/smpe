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
    pmap(args...) = map(args...)
    macro sync(ex)
        return :( $(esc(ex)) )
    end
    macro distributed(ex)
        return :( $(esc(ex)) )
    end
end

export DynamicGame, SMPEOptions, compute_equilibrium

abstract type DynamicGame end

@Base.kwdef struct SMPEOptions
    eps_outer::Real = 1e-4
    eps_inner::Real = 1e-4
    eps_contraction::Real = 1e-4
    optim_options::Optim.Options = Optim.Options()
    state_sim_eps::Real = 1e-2
    state_sim_min_sims::Integer = 100
    contraction_batch_size::Integer = 1000
    integration_rel_tol::Real = 1e-1
    integration_abs_tol::Real = 1e-3
end

DEFAULT_OPTIONS = SMPEOptions()

function compute_equilibrium(
    game::DynamicGame,
    states::Union{AbstractRange, Vector{<:AbstractRange}},
    x0::Union{Nothing, AbstractVector{<:AbstractVector{<:Number}}, AbstractArray{<:Number, 2}}=nothing,
    options::SMPEOptions=DEFAULT_OPTIONS;
    return_interpolated=true
)
    if isa(states, AbstractRange)
        states = [states]
    end
    if size(states, 1) != dim_state(game)
        throw(ArgumentError("State nodes must be of dimension dim_state=$(dim_state(game))"))
    end
    n_states = prod(length(s) for s in states)
    ALLSTATES = fill(:, dim_state(game))

    calc_policy_functions = create_init_actions(game, states, x0)
    interp_policy_functions = map(
        i -> interpolate_policy_function(
            game,
            states,
            calc_policy_functions[i]
        ),
        1:num_players(game)
    )
    calc_value_functions = initialize_value_functions(game, states)
    interp_value_functions = map(
        i -> interpolate_value_function(
            game,
            states,
            calc_value_functions[i, ALLSTATES...]
        ),
        1:num_players(game)
    )
    prev_value_functions =  copy(calc_value_functions)  # It only matters that we get the shape right for now
    prev_optimal = fill(false, num_players(game))
    attention = fill(true, (num_players(game), dim_state(game)))

    converged = fill(false, num_players(game))
    while !all(converged)
        for player_ind = 1:num_players(game)
            # calc_policy_functions and interp_policy_functions get modified in place
            calc_value_functions[player_ind, ALLSTATES...], interp_value_functions[player_ind], attention[player_ind, :] = innerloop_for_player!(
                game,
                states,
                player_ind,
                interp_value_functions[player_ind],
                calc_policy_functions,
                interp_policy_functions,
                prev_optimal[player_ind],
                options
            )

            if prev_optimal[player_ind]  # At least one iteration done before => can check convergence
                vf_norm = value_function_norm(
                    game,
                    calc_value_functions[player_ind, ALLSTATES...],
                    prev_value_functions[player_ind, ALLSTATES...]
                )
                @info "Outer loop step for player $(player_ind), step = $(vf_norm)"
                if vf_norm < options.eps_outer
                    converged[player_ind] = true
                else
                    # The actions for one player have changed. As a
                    # result, we can no longer assume that the
                    # actions calculated for other players are optimal.
                    converged = fill(false, num_players(game))
                end
            end

            # After the first iteration, the values from the previous iteration
            # are probably good starting values for future iterations. Communicate
            # this to innerloop_for_player().
            prev_optimal[player_ind] = true

            prev_value_functions[player_ind, ALLSTATES...] = calc_value_functions[player_ind, ALLSTATES...]
        end
    end

    if return_interpolated
        return interp_policy_functions, interp_value_functions, attention
    else
        return calc_policy_functions, interp_value_functions, attention
    end
end

function create_init_actions(game::DynamicGame, states, x0)
    if isnothing(x0)
        return map(
            i -> zeros((length(s) for s in states)..., num_actions(game, i)),
            1:num_players(game)
        )
    else
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

initialize_value_functions(game::DynamicGame, states) = zeros(
    num_players(game),
    (length(state_base) for state_base in states)...
)

function innerloop_for_player!(
    game::DynamicGame,
    states,
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
    state_dims = (length(s) for s in states)

    @info "Inner loop for player $(player_ind)"

    while true
        @info "Calculating attention vector..."
        if attention_cost(game, player_ind) > 0
            default_state = collect(compute_default_state(game, player_ind))
            attention = calculate_attention(
                game,
                states,
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
        if all(attention)
            # The agent pays attention to the full state. This means that we
            # do not interpolation to predict the other agents' actions as the
            # relevant states will be precisely the elements in states.
            relevant_states = Iterators.product(states...)
            calc = pmap(
                state_ind -> calculate_optimal_actions(
                    game,
                    collect(relevant_states[state_ind]),
                    player_ind,
                    interp_value_function,
                    prev_optimal ? Slices(calc_policy_functions[player_ind], dim_state(game) + 1)[state_ind] : nothing,
                    [Slices(pf, dim_state(game) + 1)[state_ind] for pf in calc_policy_functions[1:end .!= player_ind]],
                    options
                ),
                # cartesian_states supports fast linear indexing, while calc_policy_functions
                # does not. Hence we get the indices from calc_policy_functions
                # to get a CartesianIndex
                eachindex(Slices(calc_policy_functions[1], dim_state(game) + 1))
            )
        else
            # Substitute default state for actual state if agent does not pay attention
            relevant_states = Iterators.product((
                attention[i] ? states[i] : [default_state[i]]
                for i in 1:dim_state(game)
            )...)
            calc = pmap(
                state -> calculate_optimal_actions(
                    game,
                    collect(state),
                    player_ind,
                    interp_value_function,
                    prev_optimal ? eval_policy_function(interp_policy_functions[player_ind], state) : nothing,
                    [eval_policy_function(pf, state) for pf in interp_policy_functions[1:end .!= player_ind]],
                    options
                ),
                relevant_states
            )
            # Repeat optimal policies across states that the agent does not pay attention to
            calc = repeat(calc; outer=[attention[i] ? 1 : length(states[i]) for i in 1:dim_state(game)])
        end
        # We now have an Array{Array{..., 1}, n_dims}, but need a single
        # Array{..., n_actions + 1}. JuliennedArrays.Align() achieves this.
        calc_policy_functions[player_ind] = Align(calc, fill(False(), dim_state(game))..., True())
        interp_policy_functions[player_ind] = interpolate_policy_function(
            game, states, calc_policy_functions[player_ind]
        )

        @info "Contraction mapping..."
        calc_value_func, interp_value_func = calculate_value_function(
            game,
            states,
            relevant_states,
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
    states,
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
        insert_actions_at(default_actions, actions_others, player_ind)
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
        insert_actions_at(x[dim_state(game)+1:end], actions_others, player_ind)
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
    while diff > options.state_sim_eps && n_iters < options.state_sim_min_sims
        actions = map(
            pfs_player -> eval_policy_function(pfs_player, state),
            interp_policy_funcs
        )
        actions[player_ind] = default_actions
        # TODO: Fix RNG?
        state = [
            isa(s, Sampleable) ? rand(s) : s
            for s in compute_next_state(game, state, player_ind, actions)
        ]
        for i in eachindex(states)
            push!(states[i], state[i])
        end
        new_var = [var(s) for s in states]
        diff = norm(new_var - prev_var)
        prev_var = new_var
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
        insert_actions_at(x[1:n_actions], actions_others, player_ind)
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
        interp_value_function,
        next_state,
        options)
    value_func_hess = value_function_hessian_for_state(
        interp_value_function,
        next_state,
        options)#[deterministic_states, deterministic_states]

    full_state = Vector{Union{deterministic_el_type, ForwardDiff.Dual}}(undef, length(state))
    full_state[deterministic_states] = state[deterministic_states]
    stochastic_states_hess = ForwardDiff.jacobian(
        state_stochastic -> begin
            full_state[.!deterministic_states] = state_stochastic
            next_state = compute_next_state(game, state, player_ind, insert_actions_at(actions_state, actions_others, player_ind))
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
    actions_all = isnothing(actions_state) ? nothing : insert_actions_at(actions_state, actions_others, player_ind)
    # The collect is necessary because of some JuliennedArrays magic in
    # innerloop_for_player!, which can cause a type mismatch in optimization.
    # collect(...) makes sure we have an Array{T, 1} here.
    x0 = collect(compute_optimization_x0(game, state, player_ind, interp_value_function, actions_all))
    bounds = compute_action_bounds(game, state, player_ind, interp_value_function, actions_all)
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
            Fminbox(LBFGS()),
            options.optim_options
        )
    else
        # Unconstrained optimazation
        results = optimize(
            x -> calculate_negative_payoff(game, state, player_ind, interp_value_function, x, actions_others, options),
            (g, x) -> calculate_negative_payoff_gradient!(game, state, player_ind, interp_value_function, x, actions_others, g, options),
            x0,
            LBFGS(),
            options.optim_options
        )
    end
    if !Optim.converged(results)
        throw("Optimiation did not converge")
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
    actions_all = insert_actions_at(actions, actions_others, player_ind)
    payoff_now = static_payoff(game, state, player_ind, actions_all)
    next_state = compute_next_state(game, state, player_ind, actions_all)
    payoff = payoff_now + (
        discount_factor(game, player_ind) *
        value_function_for_state(interp_value_function, next_state, options)
    )
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
    actions_all = insert_actions_at(actions, actions_others, player_ind)
    payoff_now_grad = ForwardDiff.gradient(
        x -> static_payoff(
            game,
            state,
            player_ind,
            insert_actions_at(x, actions_others, player_ind)
        ), actions
    )
    next_state = compute_next_state(game, state, player_ind, actions_all)
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
    out[:] = -1 * grad
end

insert_actions_at(actions, actions_others, player_ind) = vcat(
    actions_others[1:player_ind-1], [actions], actions_others[player_ind:end]
)

interpolate_value_function(game::DynamicGame, states, calc_value_function) = interpolate_function(game, states, calc_value_function)

interpolate_policy_function(game::DynamicGame, states, calc_policy_function) = [
    interpolate_function(game, states, calc_policy_function[fill(:, length(states))..., i])
    for i in 1:size(calc_policy_function, length(states) + 1)
]

function interpolate_function(game::DynamicGame, states, vals)
    itp = interpolate(vals, BSpline(Quadratic(Free(OnGrid()))))
    # We allow extrapolation to prevent errors when the due to floating point
    # rounding we are just outside the grid. It is not advised to try to
    # extrapolate much beyond that
    return extrapolate(Interpolations.scale(itp, states...), Interpolations.Flat())
end

function calculate_value_function(
    game::DynamicGame,
    states,
    relevant_states,
    attention,
    player_ind,
    interp_value_function,
    interp_policy_functions::AbstractVector{<:AbstractVector{<:AbstractInterpolation{T, N, IT}}},
    options::SMPEOptions
) where {N, T, IT}
    # calc_value_function = SharedArray{T, N}((length(s) for s in states)...)
    calc_value_function = nothing
    prev_value_function = nothing
    # cartesian_states = map(
    #     state -> perceived_state(game, state, attention, default_state),
    #     Iterators.product(states...)
    # )

    # Cache static payoffs and state transitions
    actions_grid = map(
        pf -> map(
            state -> eval_policy_function(pf, state),
            relevant_states
        ),
        interp_policy_functions
    )
    static_payoff_grid = pmap(
        zipped -> static_payoff(game, zipped[1], player_ind, zipped[2:end]),
        zip(relevant_states, actions_grid...)
    )
    next_state_grid = pmap(
        zipped -> compute_next_state(game, zipped[1], player_ind, zipped[2:end]),
        zip(relevant_states, actions_grid...)
    )
    actions_grid = nothing  # Allow the GC to take care of this

    repeat_calc_vf = [
        indim == 1 ? outdim : 1
        for (indim, outdim)
        in zip(size(relevant_states), (length(s) for s in states))
    ]

    while true
        calc_value_function = pmap(
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
            ), Iterators.partition(
                zip(static_payoff_grid, next_state_grid),
                options.contraction_batch_size
            )
        )
        calc_value_function = reshape(
            vcat(calc_value_function...),
            size(relevant_states)
        )
        calc_value_function = repeat(calc_value_function, repeat_calc_vf...)

        #
        # @sync @distributed for (i, state) in enumerate(cartesian_states)
        #     calc_value_function[i] = calculate_value(
        #         game,
        #         perceived_state(game, state, attention, default_state),
        #         player_ind,
        #         interp_value_function,
        #         interp_policy_functions
        #     )
        # end
        if !isnothing(prev_value_function)
            diff = value_function_norm(game, calc_value_function, prev_value_function)
            @debug "Contraction mapping step = $(diff)"
            if diff < options.eps_contraction
                break
            end
        end
        interp_value_function = interpolate_value_function(game, states, calc_value_function)
        prev_value_function = copy(calc_value_function)
    end
    # return sdata(calc_value_function), interp_value_function
    return calc_value_function, interp_value_function
end

perceived_state(game::DynamicGame, state, attention, default_state) = map(
    (s, att, ds) -> att ? s : ds,
    state,
    attention,
    default_state
)

calculate_value(
    game::DynamicGame,
    next_state,
    player_ind,
    interp_value_function,
    static_payoff,
    options::SMPEOptions
) = (
    static_payoff +
    discount_factor(game, player_ind) *
    value_function_for_state(
        interp_value_function,
        next_state,
        options
    )
)

function value_function_for_state(
    interp_value_function::AbstractInterpolation{T, N, NT},
    state::AbstractVector{<:Real},
    options::SMPEOptions
)::T where {T<:Real, N, NT}
    interp_value_function(state...)
end

function value_function_for_state(
    interp_value_function::AbstractInterpolation{T, N, NT},
    state::AbstractVector{<:Union{<:Real, ContinuousUnivariateDistribution}},
    options::SMPEOptions
)::T where {T<:Real, N, NT}
    deterministic_states = map(s -> isa(s, T), state)
    return integrate_fn_over_stochastic_states(
        # We need an Array here since that is what is needed for numerical integration
        x -> [value_function_for_state(interp_value_function, x, options)],
        1,
        interpolation_bounds(interp_value_function),
        state,
        deterministic_states,
        options
    )[1]
end

function value_function_gradient_for_state(
    interp_value_function::AbstractInterpolation{T, N, NT},
    state::AbstractVector{<:Real},
    options::SMPEOptions
)::Vector{T} where {T<:Real, N, NT}
    Interpolations.gradient(interp_value_function, state...)
end

function value_function_gradient_for_state(
    interp_value_function::AbstractInterpolation{T, N, NT},
    state::AbstractVector{<:Union{Real, ContinuousUnivariateDistribution}},
    options::SMPEOptions
)::Vector{T} where {T<:Real, N, NT}
    deterministic_states = map(s -> isa(s, T), state)
    return integrate_fn_over_stochastic_states(
        x -> value_function_gradient_for_state(interp_value_function, x, options)[deterministic_states],
        sum(deterministic_states),
        interpolation_bounds(interp_value_function),
        state,
        deterministic_states,
        options
    )
end

function value_function_gradient_for_actions(
    game::DynamicGame,
    state::AbstractVector{T},
    next_state::AbstractVector{<:Real},
    player_ind,
    interp_value_function::AbstractInterpolation{T, N, NT},
    actions,
    actions_others,
    options::SMPEOptions
)::Vector{T} where {T<:Real, N, NT}
    next_state_jac = ForwardDiff.jacobian(
        x -> compute_next_state(
            game,
            state,
            player_ind,
            insert_actions_at(x, actions_others, player_ind)
        ),
        actions
    )
    next_vf_grad = value_function_gradient_for_state(interp_value_function, next_state, options)
    Interpolations.gradient(interp_value_function, state...)
    return transpose(next_state_jac) * next_vf_grad
end

function value_function_gradient_for_actions(
    game::DynamicGame,
    state::AbstractVector{T},
    next_state::AbstractVector{<:Union{<:Real, ContinuousUnivariateDistribution}},
    player_ind,
    interp_value_function::AbstractInterpolation{T, N, NT},
    actions,
    actions_others,
    options::SMPEOptions
)::Vector{T} where {T<:Real, N, NT}
    # Only differentiate wrt the non-stochastic parts of the state
    # (i.e. it is assumed that the stochastic state transitions are exogenous)
    deterministic_states = map(s -> isa(s, T), next_state)
    # We can hence take the Jacobian of the state evolution wrt to
    # the deterministic states only
    next_state_jac = ForwardDiff.jacobian(
        x -> calculate_next_state_deterministic_part(
            deterministic_states,
            game,
            state,
            player_ind,
            insert_actions_at(x, actions_others, player_ind)
        ),
        actions
    )

    # Integrate out the random variables in the gradient of the value function
    # Integration needs [0, 1] box => use a change of variable
    next_vf_grad = integrate_fn_over_stochastic_states(
        x -> value_function_gradient_for_state(interp_value_function, x, options)[deterministic_states],
        sum(deterministic_states),
        interpolation_bounds(interp_value_function),
        next_state,
        deterministic_states,
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
    fn,
    dim_fn,
    domain_fn,
    state,
    deterministic_states,
    options::SMPEOptions
)
    stochastic_states = state[.!deterministic_states]
    change_var = map(
        x -> make_scaler(x...),
        zip(stochastic_states, domain_fn[.!deterministic_states])
    )
    integrand_state = [
        is_deterministic ? x : NaN
        for (is_deterministic, x) in
        zip(deterministic_states, state)
    ]
    integral = vegas(
        (x, f) -> begin
            integrand_state[.!deterministic_states] = [
                cv[1](y) for (cv, y) in zip(change_var, x)
            ]
            cv_jac = prod(cv[2](y) for (cv, y) in zip(change_var, x))
            weight = prod(pdf(s, y) for (s, y) in zip(stochastic_states, x))
            f[:] = fn(integrand_state) * cv_jac * weight
        end,
        length(stochastic_states),
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
    interp_value_function::AbstractInterpolation{T, N, NT},
    states::AbstractVector{T},
    options::SMPEOptions
)::Matrix{T} where {T<:Real, N, NT}
    if ndims(interp_value_function) <= 3
        return Interpolations.hessian(interp_value_function, states...)
    else
        # There is a bug in Interpolations.jl for computing the Hssian when
        # ndim >= 4. See https://github.com/JuliaMath/Interpolations.jl/issues/364
        return ForwardDiff.hessian(x -> interp_value_function(x...), states)
    end
end

function value_function_hessian_for_state(
    interp_value_function::AbstractInterpolation{T, N, NT},
    state::AbstractVector{<:Union{Real, ContinuousUnivariateDistribution}},
    options::SMPEOptions
)::Matrix{T} where {T<:Real, N, NT}
    deterministic_states = map(s -> isa(s, T), state)
    n_deterministic = sum(deterministic_states)
    hess_vectorized = integrate_fn_over_stochastic_states(
        # Integration only possible over vectors. Hence vectorize and reshape below
        x -> vec(value_function_hessian_for_state(interp_value_function, x, options)[deterministic_states, deterministic_states]),
        n_deterministic^2,
        interpolation_bounds(interp_value_function),
        state,
        deterministic_states,
        options
    )
    return reshape(hess_vectorized, n_deterministic, n_deterministic)
end

function interpolation_bounds(interp::AbstractInterpolation)
    vf_bounds = bounds(interp)
    return collect(zip(
        map(minimum, vf_bounds),
        map(maximum, vf_bounds)))
end

# Extrapolated functions have no bounds, give the bounds of the
# underlying interpolated function
interpolation_bounds(interp::AbstractExtrapolation) = interpolation_bounds(parent(interp))

compute_optimization_x0(
    game::DynamicGame, state, player_ind, interp_value_function, actions
) = isnothing(actions) ? zeros(num_actions(game, player_ind)) : actions[player_ind]

compute_action_bounds(
    game::DynamicGame, state, player_ind, interp_value_function, actions
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
static_payoff(g::DynamicGame, state, player_ind, actions) = throw("Concrete subtypes of DynamicGame must implement discount_factor()")
compute_next_state(g::DynamicGame, state, player_ind, actions) = throw("Concrete subtypes of DynamicGame must implement compute_next_state()")
attention_cost(g::DynamicGame, player_ind) = throw("Concrete subtypes of DynamicGame must implement attention_cost()")
compute_default_state(g::DynamicGame, player_ind) = throw("Concrete subtypes of DynamicGame must implement compute_default_state() when attention costs are positive")

end
