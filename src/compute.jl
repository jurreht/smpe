struct OptimizationConvergenceError <: Exception
    opt_results::Optim.OptimizationResults
end
Base.showerror(io::IO, e::OptimizationConvergenceError) = print(io, "optimization did not converge")

mutable struct SolutionProgress
    calc_policy_functions::Vector{Array{Float64}}
    calc_value_functions::Vector{Array{Float64}}
    prev_value_functions::Vector{Array{Float64}}
    prev_optimal::Vector{Bool}
    attention::Matrix{Float64}
    converged::Vector{Bool}
    last_player::Int
    compute_time::Float64
end

function compute_equilibrium(
    game::DynamicGame,
    nodes::Union{AbstractRange, Vector{<:AbstractRange}};
    x0=nothing,
    options::SMPEOptions=DEFAULT_OPTIONS,
    progress_cache::Union{IO, AbstractString, Nothing}=nothing,
    fix_attention::Union{<:AbstractMatrix{Float64}, Nothing}=nothing
)
    if isa(nodes, AbstractRange)
        nodes = [nodes]
    end
    if size(nodes, 1) != dim_rectangular_state(game)
        throw(ArgumentError("State nodes must be of dimension dim_rectangular_state=$(dim_rectangular_state(game))"))
    end
    n_nodes = prod(length(s) for s in nodes)

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
        att0 = isnothing(fix_attention) ? fill(1.0, (num_players(game), dim_state(game))) : fix_attention
        progress = SolutionProgress(
            create_init_actions(game, nodes, x0),
            initialize_value_functions(game, nodes),
            initialize_value_functions(game, nodes),  # It only matters that we get the shape right for now
            fill(isnothing(x0) ? false : true, num_players(game)),
            att0,
            fill(false, num_players(game)),
            0,
            0.
        )
    end

    interp_policy_functions = Vector{Vector{SMPEInterpolation}}(undef, num_players(game))
    interp_value_functions = Vector{SMPEInterpolation}(undef, num_players(game))
    Threads.@threads for player_ind in 1:num_players(game)
        # For first interpolatipn, we just attention to zero everywhere.
        # This leads to the fastest initilaziation
        interp_policy_functions[player_ind] = interpolate_policy_function(
            game,
            nodes,
            zeros(dim_rectangular_state(game)),
            progress.calc_policy_functions[player_ind],
            options
        )
        interp_value_functions[player_ind] = interpolate_value_function(
            game,
            nodes,
            zeros(dim_rectangular_state(game)),
            progress.calc_value_functions[player_ind],
            options
        )
    end

    # Pre-initialize value function if starting x were passed
    # Only do this if we started from a fresh progress cache, otherwise it
    # is wasted effort
    if !isnothing(x0) && progress.compute_time == 0
        @info "Initial contraction mapping"
        for i in 1:num_players(game)
            progress.compute_time += @elapsed begin
                progress.calc_value_functions[i], interp_value_functions[i] = calculate_value_function(
                    game,
                    nodes,
                    Iterators.product(nodes...),
                    fill(1.0, dim_state(game)),
                    zeros(dim_state(game)),  # Irrelvant given attention = 1
                    i,
                    interp_value_functions[i],
                    interp_policy_functions,
                    options
                )
            end
        end

        if !isnothing(progress_cache)
            serialize(progress_cache, progress)
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
                progress.calc_value_functions[player_ind], interp_value_functions[player_ind], progress.attention[player_ind, :] = innerloop_for_player!(
                    game,
                    nodes,
                    player_ind,
                    interp_value_functions[player_ind],
                    progress.calc_policy_functions,
                    interp_policy_functions,
                    progress.prev_optimal[player_ind],
                    fix_attention,
                    options
                )

                if progress.prev_optimal[player_ind]  # At least one iteration done before => can check convergence
                    vf_norm = value_function_norm(
                        game,
                        progress.calc_value_functions[player_ind],
                        progress.prev_value_functions[player_ind]
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

                progress.prev_value_functions[player_ind] = progress.calc_value_functions[player_ind]

                progress.last_player = player_ind < num_players(game) ? player_ind : 0
            end

            if !isnothing(progress_cache)
                serialize(progress_cache, progress)
            end
        end
    end

    return Equilibrium(
        interp_policy_functions,
        interp_value_functions,
        progress.attention,
        progress.compute_time
    )
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

initialize_value_functions(game::DynamicGame, nodes) = [
    zeros((length(node_base) for node_base in nodes)...)
    for i in 1:num_players(game)]

function innerloop_for_player!(
    game::DynamicGame,
    nodes,
    player_ind,
    interp_value_function,
    calc_policy_functions,
    interp_policy_functions,
    prev_optimal,
    fix_attention,
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
        if attention_cost(game, player_ind) > 0 && isnothing(fix_attention)
            default_state = collect(compute_default_state(game, player_ind))
            attention = calculate_attention(
                game,
                default_state,
                player_ind,
                interp_value_function,
                interp_policy_functions,
                options
            )
        elseif !isnothing(fix_attention)
            default_state = collect(compute_default_state(game, player_ind))
            attention = fix_attention[player_ind, :]
        else
            default_state = fill(0.0, dim_state(game))
            attention = fill(1.0, dim_state(game))
        end
        @debug "Attention vector = $(attention)"

        @info "Calculating policy function..."
        # Substitute default state for actual state if agent does not pay attention
        default_node = transform_state_back(game, default_state)
        relevant_nodes = Iterators.product((
            attention[i] > options.attention_cutoff ? nodes[i] : [default_node[i]]
            for i in 1:dim_rectangular_state(game)
        )...)
        calc = Array{Float64}(undef, size(relevant_nodes)..., num_actions(game, player_ind))
        inds = eachindex(Slices(calc, dim_rectangular_state(game) + 1))
        Threads.@threads for node_ind in inds
            state = perceived_state(
                game,
                collect(transform_state(game, relevant_nodes[node_ind])),
                attention,
                default_state
            )
            calc[node_ind, :] = calculate_optimal_actions(
                game,
                state,
                player_ind,
                interp_value_function,
                prev_optimal ? eval_policy_function(interp_policy_functions[player_ind], state) : nothing,
                [eval_policy_function(pf, state) for pf in interp_policy_functions[1:end .!= player_ind]],
                options
            )
        end
        calc_policy_functions[player_ind] = calc
        interp_policy_functions[player_ind] = interpolate_policy_function(
            game, nodes, attention, calc_policy_functions[player_ind], options
        )

        @info "Contraction mapping..."
        calc_value_func, interp_value_func = calculate_value_function(
            game,
            nodes,
            relevant_nodes,
            attention,
            default_state,
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
    interp_policy_funcs,
    options::SMPEOptions
)
    @debug "Default state $(default_state)"
    benefit_attention = calculate_benefit_attention(
        game,
        default_state,
        player_ind,
        interp_value_function,
        interp_policy_funcs,
        options
    )
    att_cost = attention_cost(game, player_ind)
    return [attention_function(game, b / att_cost) for b in benefit_attention]
end

function static_benefit_attention(
    game::DynamicGame,
    player_ind,
    default_state,
    actions_player,
    actions_others,
    state_ind
)
    padded_actions_others = pad_actions(game, actions_others, player_ind)
    constant_state = vcat(default_state[1:state_ind-1], default_state[1+state_ind:end])
    total_hessian = ForwardDiff.hessian(
        x -> static_payoff(
            game,
            vcat(constant_state[1:state_ind-1], x[1], constant_state[state_ind:end]), # state
            player_ind,
            x[2:end],  # actions_player
            padded_actions_others
        ),
        vcat(default_state[state_ind], actions_player)
    )
    d_action_state = total_hessian[2:end, 1]
    d_action_action = total_hessian[2:end, 2:end]
    return -1 * dot(d_action_state, d_action_action \ d_action_state)
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
        pad_actions(game, actions_others, player_ind)
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

function calculate_optimal_actions(
    game::DynamicGame,
    state,
    player_ind,
    interp_value_function,
    actions_state,
    actions_others,
    options::SMPEOptions,
    dynamic=true
)
    padded_actions = pad_actions(game, actions_others, player_ind)
    # The collect is necessary because of some JuliennedArrays magic in
    # innerloop_for_player!, which can cause a type mismatch in optimization.
    # collect(...) makes sure we have an Array{T, 1} here.
    x0 = collect(compute_optimization_x0(game, state, player_ind, interp_value_function, actions_state, padded_actions))
    bounds = compute_action_bounds(game, state, player_ind, interp_value_function, actions_state, padded_actions)
    return maximize_payoff(game, state, player_ind, interp_value_function, actions_others, x0, bounds, options, dynamic)
end

function maximize_payoff(
    game::DynamicGame,
    state,
    player_ind,
    interp_value_function,
    actions_others,
    x0,
    bounds,
    options::SMPEOptions,
    dynamic
)
    lb = [isnothing(bounds[i][1]) ? -Inf : bounds[i][1] for i in eachindex(bounds)]
    ub = [isnothing(bounds[i][2]) ? Inf : bounds[i][2] for i in eachindex(bounds)]

    if any(-Inf .< lb) || any(Inf .> ub)
        # Constrained optimization

        # Make sure starting value is withing bounds
        for i in 1:length(x0)
            if x0[i] < lb[i] || x0[i] > ub[i]
                if isinf(lb[i])
                    x0[i] = ub[i] - 1
                elseif isinf(ub[i])
                    x0[i] = lb[i] + 1
                else
                    x0[i] = (ub[i] + lb[i]) / 2
                end
            end
        end

        results = optimize(
            x -> calculate_negative_payoff(game, state, player_ind, interp_value_function, x, actions_others, options, dynamic),
            (g, x) -> calculate_negative_payoff_gradient!(game, state, player_ind, interp_value_function, x, actions_others, g, options, dynamic),
            lb,
            ub,
            x0,
            Fminbox(options.optim_method),
            options.optim_options
        )
    else
        # Unconstrained optimazation
        results = optimize(
            x -> calculate_negative_payoff(game, state, player_ind, interp_value_function, x, actions_others, options, dynamic),
            (g, x) -> calculate_negative_payoff_gradient!(game, state, player_ind, interp_value_function, x, actions_others, g, options, dynamic),
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
    options::SMPEOptions,
    dynamic::Bool
)
    padded_actions = pad_actions(game, actions_others, player_ind)
    payoff_now = static_payoff(game, state, player_ind, actions, padded_actions)
    delta = discount_factor(game, player_ind)
    if delta > 0 && dynamic
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
    options::SMPEOptions,
    dynamic::Bool
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
    if delta > 0 && dynamic
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

function calculate_value_function(
    game::DynamicGame,
    nodes,
    relevant_nodes,
    attention,
    default_state,
    player_ind,
    interp_value_function,
    interp_policy_functions,
    options::SMPEOptions
)
    calc_value_function = nothing
    prev_value_function = nothing

    # Cache static payoffs and state transitions
    relevant_states = map(node -> transform_state(game, collect(node)), relevant_nodes)
    perceived_states = map(state -> perceived_state(game, state, attention, default_state), relevant_states)
    size_states = size(relevant_states)
    state_inds = Iterators.product([1:s for s in size_states]...)
    actions_grid = Vector{Array{Float64, dim_rectangular_state(game) + 1}}()
    for i in 1:num_players(game)
        pf = interp_policy_functions[i]
        grid_player = Array{Float64}(undef, size_states..., num_actions(game, i))
        Threads.@threads for state_ind in state_inds
            # For the current player, we evaluate the policy function in the true
            # state, but for the other players in the current player's perceived
            # state. This is because the way the policy functions are
            # interpolated. The actual policy function of player_ind in a certain
            # state is calculated taking into account inattention. For the other
            # players, however the current player expects pf_j(perceived_i(state)),
            # where pf_j is j's objective policy function.
            state = i == player_ind ? relevant_states[state_ind...] : perceived_states[state_ind...]
            grid_player[state_ind..., :] = eval_policy_function(pf, state)
        end
        push!(actions_grid, grid_player)
    end
    static_payoff_grid = Array{Float64}(undef, size_states...)
    first_state = iterate(state_inds)[1]
    # It is important we get this right, because it will be used for multiple
    # dispatch when calculating the value function
    next_state_type = typeof(
        compute_next_state(
            game,
            relevant_states[first_state...],
            player_ind,
            actions_grid[player_ind][first_state..., :],
            [actions_grid[i][first_state..., :] for i in 1:num_players(game)]
        )
    ).parameters[1]
    next_state_grid = Array{next_state_type}(undef, size_states..., dim_state(game))
    Threads.@threads for state_ind in state_inds
        # Here, we do need the perdeived state since this what the agent perceives
        # happens.
        state = perceived_state(
            game,
            relevant_states[state_ind...],
            attention,
            default_state
        )
        actions = [actions_grid[i][state_ind..., :] for i in 1:num_players(game)]
        static_payoff_grid[state_ind...] = static_payoff(
            game, state, player_ind, actions[player_ind], actions
        )
        next_state_grid[state_ind..., :] = compute_next_state(
            game, state, player_ind, actions[player_ind], actions
        )
    end
    # Allow the GC to take care of this
    relevant_states = nothing
    actions_grid = nothing

    while true
        calc_value_function = Array{Float64}(undef, size_states...)
        Threads.@threads for state_ind in state_inds
            calc_value_function[state_ind...] = calculate_value(
                game,
                next_state_grid[state_ind..., :],
                player_ind,
                interp_value_function,
                static_payoff_grid[state_ind...],
                options
            )
        end
        # repeat_calc_vf = [
        #     indim == 1 ? outdim : 1
        #     for (indim, outdim)
        #     in zip(size(relevant_nodes), (length(s) for s in nodes))
        # ]
        # calc_value_function = repeat(calc_value_function, repeat_calc_vf...)

        if !isnothing(prev_value_function)
            diff = value_function_norm(game, calc_value_function, prev_value_function)
            @debug "Contraction mapping step = $(diff)"
            if diff < options.eps_contraction
                break
            end
        end
        interp_value_function = interpolate_value_function(game, nodes, attention, calc_value_function, options)
        prev_value_function = copy(calc_value_function)
    end
    return calc_value_function, interp_value_function
end

# Convergence criterion as in Doraszelski & Pakes (2007)
value_function_norm(game::DynamicGame, new_value, old_value) = max(
    map(abs, (new_value .- old_value) ./ (1 .+ new_value))...
)
