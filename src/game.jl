abstract type DynamicGame end

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
attention_function(g::DynamicGame, sigma2) = sigma2 >= 2 ? 1.0 : 0.0
dim_rectangular_state(g::DynamicGame) = dim_state(g)
transform_state(g::DynamicGame, state) = state
# Note this Jacobian *should* be square. So if the transformation node -> state
# changes the number of dimensions, do not include the derivatives wrt linearly
# dependent states.
transform_state_jac(g::DynamicGame, state) = I
transform_state_back(g::DynamicGame, state) = state
transform_state_back_jac(g::DynamicGame, state) = I
transform_state_back_hessian(g::DynamicGame, state) = fill(I, dim_rectangular_state(g))

compute_optimization_x0(
    game::DynamicGame, state, player_ind, interp_value_function, actions_player, actions
) = isnothing(actions_player) ? zeros(num_actions(game, player_ind)) : actions_player

compute_action_bounds(
    game::DynamicGame, state, player_ind, interp_value_function, actions_player, actions
) = fill((nothing, nothing), num_actions(game, player_ind))

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
