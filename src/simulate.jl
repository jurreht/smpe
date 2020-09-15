function simulate_equilibrium_path(
    game::DynamicGame,
    equilibrium::Equilibrium,
    start_state::Vector{Float64},
    length::Integer;
    rng::AbstractRNG=Random.GLOBAL_RNG
)::EquilibriumPath
    path = Vector{Tuple{State, Actions}}(undef, length)
    state = copy(start_state)
    for t in 1:length
        actions = [[pf(state...) for pf in pf_player] for pf_player in equilibrium.pf]
        path[t] = (state, actions)
        # player_ind is arbitrary here, use 1 since it always exists
        state_dist = compute_next_state(game, state, 1, actions[1], actions)
        state = [isa(x, Sampleable) ? rand(rng, x) : x for x in state_dist]
    end
    return path
end

function simulate_equilibrium_values(
    game::DynamicGame,
    start_state::Vector{<:Union{ContinuousUnivariateDistribution, <:Real}},
    default_state::Vector{Float64},
    attention::Vector{S},
    player_ind::Integer,
    policy_functions::Vector{Vector{T}},
    length::Integer,
    rands::Matrix{Float64}
)::Real where {S <: Real, T <: SMPEInterpolation}
    payoff = zero(S)
    delta = [discount_factor(game, i) for i in 1:num_players(game)]
    state_dist = start_state
    for t in 1:length
        state = [isa(x, Sampleable) ? quantile(x, rands[t, i]) : x for (i, x) in enumerate(state_dist)]
        sparse_state = perceived_state(game, state, attention, default_state)
        actions = [
            [i == player_ind ? pf(state...) : pf(sparse_state...) for pf in pf_player]
            for (i, pf_player) in enumerate(policy_functions)
        ]
        payoff += delta[player_ind]^(t - 1) * static_payoff(game, sparse_state, player_ind, actions[player_ind], actions)
        state_dist = compute_next_state(game, sparse_state, player_ind, actions[player_ind], actions)
    end
    return payoff
end
