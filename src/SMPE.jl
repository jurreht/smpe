module SMPE

using Debugger

using Distributed
using ForwardDiff
using Logging
using Interpolations
using Optim
using SharedArrays

# This allows to comment away "using Logging" above and have working code
# with some messages. This is useful during development, since the Juno editor
# has some issues with finding the right line when a macro causes a multi-line
# expansion.
if !@isdefined(Logging)
    macro info(ex)
        return :( println($(esc(ex))) )
    end
end

export DynamicGame, SMPEOptions, compute_equilibrium

abstract type DynamicGame end

@Base.kwdef struct SMPEOptions
    eps_outer::Real = 1e-4
    eps_inner::Real = 1e-4
    optim_options::Optim.Options = Optim.Options()
end

DEFAULT_OPTIONS = SMPEOptions()

function compute_equilibrium(
    game::DynamicGame,
    states::Union{AbstractRange, Vector{<:AbstractRange}},
    x0::Union{Nothing, AbstractVector{<:AbstractVector{<:Number}}, AbstractArray{<:Number, 2}}=nothing,
    options::SMPEOptions=DEFAULT_OPTIONS
)
    if isa(states, AbstractRange)
        states = [states]
    end
    if size(states, 1) != dim_state(game)
        throw(ArgumentError("State nodes must be of dimension dim_state=$(dim_state(game))"))
    end
    n_states = prod(length(s) for s in states)
    ALLSTATES = fill(:, dim_state(game))

    policy_functions = create_init_actions(game, n_states, x0)
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

    converged = fill(false, num_players(game))
    while !all(converged)
        for player_ind = 1:num_players(game)
            # policy_functions gets modified in place
            calc_value_functions[player_ind, ALLSTATES...], interp_value_functions[player_ind] = innerloop_for_player(
                game,
                states,
                player_ind,
                interp_value_functions[player_ind],
                policy_functions,
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

    return policy_functions, interp_value_functions
end

function create_init_actions(game::DynamicGame, n_states, x0)
    if isnothing(x0)
        return [zeros(n_states, num_actions(game, i)) for i in 1:num_players(game)]
    else
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

function innerloop_for_player(
    game::DynamicGame,
    states,
    player_ind,
    interp_value_function,
    policy_functions,
    prev_optimal,
    options::SMPEOptions
)
    prev_value_func = nothing
    calc_value_func = nothing
    interp_value_func = nothing
    state_dims = (length(s) for s in states)

    @info "Inner loop for player $(player_ind)"

    while true
        @info "Calculating policy function..."
        cartesian_states = Iterators.product(states...)
        calc = pmap(
            state_ind -> calculate_optimal_actions(
                game,
                cartesian_states[state_ind],
                player_ind,
                interp_value_function,
                prev_optimal ? policy_functions[player_ind][state_ind, :] : nothing,
                [pf[state_ind, :] for pf in policy_functions[1:end .!= player_ind]],
                options
            ),
            eachindex(cartesian_states)
        )
        calc = [calc[i][1] for i in eachindex(calc)]
        if num_actions(game, player_ind) == 1
            # When there is one action for this player, pmap() loses a
            # dimension and this errors the assignment right below
            calc = reshape(calc, size(calc, 1), 1)
        end
        policy_functions[player_ind] = reshape(calc, (prod(state_dims), num_actions(game, player_ind)))
        @info "Contraction mapping..."
        calc_value_func, interp_value_func = calculate_value_function(
            game,
            states,
            player_ind,
            interp_value_function,
            policy_functions
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
    end

    return calc_value_func, interp_value_func
end

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
    x0 = compute_optimization_x0(game, state, player_ind, interp_value_function, actions_all)
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
            x -> calculate_negative_payoff(game, state, player_ind, interp_value_function, x, actions_others),
            (g, x) -> calculate_negative_payoff_gradient!(game, state, player_ind, interp_value_function, x, actions_others, g),
            lb,
            ub,
            x0,
            Fminbox(LBFGS()),
            options.optim_options
        )
    else
        # Unconstrained optimazation
        results = optimize(
            x -> calculate_negative_payoff(game, state, player_ind, interp_value_function, x, actions_others),
            (g, x) -> calculate_negative_payoff_gradient!(game, state, player_ind, interp_value_function, x, actions_others, g),
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

function calculate_negative_payoff(game::DynamicGame, state, player_ind, interp_value_function, actions, actions_others)
    actions_all = insert_actions_at(actions, actions_others, player_ind)
    payoff_now = static_payoff(game, state, player_ind, actions_all)
    next_state = compute_next_state(game, state, player_ind, actions_all)
    payoff = payoff_now + (
        discount_factor(game, player_ind) *
        interp_value_function(next_state...)
    )
    return -1 * payoff
end

function calculate_negative_payoff_gradient!(game::DynamicGame, state, player_ind, interp_value_function, actions, actions_others, out)
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
    next_state_jac = ForwardDiff.jacobian(
        x -> compute_next_state(
            game,
            state,
            player_ind,
            insert_actions_at(x, actions_others, player_ind)
        ),
        actions
    )
    next_vf_grad = Interpolations.gradient(interp_value_function, next_state...)
    grad = payoff_now_grad + discount_factor(game, player_ind) * transpose(next_state_jac) * next_vf_grad
    out[:] = -1 * grad
end

insert_actions_at(actions, actions_others, player_ind) = vcat(
    actions_others[1:player_ind-1], [actions], actions_others[player_ind:end]
)

function interpolate_value_function(game::DynamicGame, states, calc_value_functions)
    itp = interpolate(calc_value_functions, BSpline(Quadratic(Free(OnGrid()))))
    return scale(itp, states...)
end

function calculate_value_function(game::DynamicGame, states, player_ind, interp_value_function, policy_functions::AbstractVector{<:AbstractArray{T, 2}}) where T
    calc_value_function = SharedArray{T}((length(s) for s in states)...)
    prev_value_function = nothing
    while true
        @sync @distributed for (i, state) in enumerate(Iterators.product(states...))
            calc_value_function[i] = calculate_value(
                game,
                state,
                player_ind,
                interp_value_function,
                [policy_functions[j][i, :] for j in eachindex(policy_functions)]
            )
        end
        if !isnothing(prev_value_function) && value_function_norm(game, calc_value_function, prev_value_function) < 1e-5
            break
        end
        interp_value_function = interpolate_value_function(game, states, calc_value_function)
        prev_value_function = copy(sdata(calc_value_function))
    end
    return sdata(calc_value_function), interp_value_function
end

calculate_value(game::DynamicGame, state, player_ind, interp_value_function, actions) = (
    static_payoff(game, state, player_ind, actions) +
    discount_factor(game, player_ind) *
    interp_value_function(compute_next_state(game, state, player_ind, actions)...)
)

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

end
