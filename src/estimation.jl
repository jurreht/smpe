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

function bracket_attention_cost(moments, att_cost_lower, att_cost_upper, attention, alpha, options)
    test_lower = test_moments(moments, att_cost_lower, attention, alpha, options)
    test_upper = test_moments(moments, att_cost_upper, attention, alpha, options)
    if test_lower && test_upper
        # Model can be rejected for full interval
        return nothing
    end
    while test_moments(moments, att_cost_lower, attention, alpha, options)
        att_cost_lower += options.att_cost_bracket_step
    end
    while test_moments(moments, att_cost_upper, attention, alpha, options)
        att_cost_upper -= options.att_cost_bracket_step
    end
    return (att_cost_lower, att_cost_upper)
end

function test_moments(moments, att_cost, attention, alpha, options::SMPEOptions)
    moments_with_att_cost = Matrix{Float64}(undef, size(moments)...)
    for m in 1:size(moments, 1)
        for state_ind in 1:size(moments, 2)
            if attention[state_ind] > options.attention_cutoff
                moments_with_att_cost[m, state_ind] = att_cost - moments[m, state_ind]
            else
                moments_with_att_cost[m, state_ind] = moments[m, state_ind] - att_cost
            end
        end
    end
    moments_mean = mean(moments_with_att_cost; dims=1)[1, :]
    moments_sd = std(moments_with_att_cost; dims=1)[1, :]
    if any(moments_mean[moments_sd .<= 1e-15] .> 0)
        return true
    end
    moments_with_att_cost = moments_with_att_cost[:, moments_sd .> 1e-15]
    return moment_inequality.test_inequalities(moments_with_att_cost, alpha, .1 * alpha)
end
