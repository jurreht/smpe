function estimate_attention_cost(
    game::DynamicGame,
    player_ind::Int,
    attention::Vector{Float64},
    path::EquilibriumPath,
    policy_functions,
    nodes,
    att_cost_range::Tuple{Float64, Float64},
    alpha::Float64,
    options::SMPEOptions=DEFAULT_OPTIONS
)::Union{Tuple{Float64, Float64}, Nothing}
    moments = attention_cost_moments(game, player_ind, attention, path, policy_functions, nodes, options)
    att_cost_lower, att_cost_upper = att_cost_range
    return bracket_attention_cost(moments, att_cost_lower, att_cost_upper, attention, alpha, options)
end

function attention_cost_moments(
    game::DynamicGame,
    player_ind::Int,
    attention::Vector{Float64},
    path::EquilibriumPath,
    policy_functions,
    nodes,
    options::SMPEOptions=DEFAULT_OPTIONS
)::Matrix{Float64}
    default_state = compute_default_state(game, player_ind)

    interp_start = interpolate_value_function(game, nodes, zeros(dim_rectangular_state(game)), zeros(length(nodes[1])), options)
    _, value_function = calculate_value_function(
        game,
        nodes,
        calculate_relevant_nodes(game, nodes, default_state, attention, options),
        attention,
        default_state,
        player_ind,
        interp_start,
        policy_functions,
        options
    )

    benefit_attention = calculate_benefit_attention(
        game,
        default_state,
        player_ind,
        value_function,
        policy_functions,
        options,
        false
    )

    # Calculate moments without attention cost
    n_moments = size(path, 1)
    moments = Matrix{Float64}(undef, n_moments, dim_state(game))
    for m in 1:n_moments
        state = path[m][1]
        for state_ind in 1:dim_state(game)
            moments[m, state_ind] = benefit_attention[state_ind] * (state[state_ind] - default_state[state_ind])^2
        end
    end
    return moments
end
