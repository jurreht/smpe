using Distributed
addprocs(4; exeflags="--project")

using Test
@everywhere using SMPE
@everywhere using Statistics

@everywhere struct CapitalAccumulationProblem{T<:Real} <: DynamicGame
	alpha::T
	beta::T
	k_steady_state::T
	k_lowest::T

	function CapitalAccumulationProblem(alpha::T, beta::T) where T<:Real
		k_steady_state = (alpha * beta) ^ (1 / (1 - alpha))
		k_lowest = 1e-5 * k_steady_state
		return new{T}(alpha, beta, k_steady_state, k_lowest)
	end
end

@everywhere SMPE.num_players(g::CapitalAccumulationProblem) = 1
@everywhere SMPE.dim_state(g::CapitalAccumulationProblem) = 1
@everywhere SMPE.num_actions(g::CapitalAccumulationProblem, i) = 1
@everywhere SMPE.discount_factor(g::CapitalAccumulationProblem, i) = g.beta
@everywhere SMPE.static_payoff(
	game::CapitalAccumulationProblem,
	state,
	player_ind,
	actions
) = log(state[1] ^ game.alpha - actions[1][1])
@everywhere SMPE.compute_next_state(
	game::CapitalAccumulationProblem,
	state,
	player_ind,
	actions
) = actions[1]
@everywhere SMPE.compute_action_bounds(
	game::CapitalAccumulationProblem,
	state,
	player_ind,
	interp_value_function,
	actions
) = [(game.k_lowest, state[1] ^ game.alpha)]
@everywhere SMPE.compute_optimization_x0(
	game::CapitalAccumulationProblem,
	state,
	player_ind,
	interp_value_function,
	actions
) = isnothing(actions) ? [state[1] ^ game.alpha / 2] : actions[1]

alpha = 0.5
beta = 0.95
game = CapitalAccumulationProblem(alpha, beta)
grid = range(game.k_lowest, 4*game.k_steady_state, length=100)
pf, vf = compute_equilibrium(game, grid)
policy_exact = alpha * beta * collect(grid) .^ alpha
@test pf[1] ≈ policy_exact atol=1e-2 rtol=5e-1

@everywhere struct SwitchingModel <: DynamicGame
	n_firms
	marginal_cost
	switch_cost
    new_consumers
    df_firms
    df_consumers
end

@everywhere SMPE.num_players(game::SwitchingModel) = game.n_firms
@everywhere SMPE.dim_state(game::SwitchingModel) = game.n_firms
@everywhere SMPE.num_actions(game::SwitchingModel, i) = 1
@everywhere SMPE.discount_factor(game::SwitchingModel, i) = game.df_firms
@everywhere function SMPE.static_payoff(
	game::SwitchingModel,
	state,
	player_ind,
	actions
)
    pi = actions[player_ind][1]
    p_other = mean(x[1] for x in actions[1:end .!= player_ind])
    xi = state[player_ind]
    ci = game.marginal_cost[player_ind]

    # eq. 10
    demand_old = (
        0.5
        * (game.n_firms - 1)
        * ((1 - pi + p_other) + (game.n_firms * xi - 1) * game.switch_cost)
    )

    # eq. 12
    demand_young = (
        0.5 * (game.n_firms - 1) * (1 - pi + p_other) * game.new_consumers
    )

    # first part of eq. 22
    return (pi - ci) * (demand_young + demand_old)
end
@everywhere function SMPE.compute_next_state(
	game::SwitchingModel,
	state,
	player_ind,
	actions,
)
	# When there is automatic differentiation, actions[player_ind] may be some
	# kind of DualNumber. Then zeros() need to create an array of the same type
	# otherwise we will not be able to assign to next_state below.
	state_type = typeof(actions[player_ind]).parameters[1]
    next_state = zeros(state_type, game.n_firms)

    for i in 1:game.n_firms
        pi = actions[i][1]
        p_other = mean(x[1] for x in actions[1:game.n_firms .!= i])

        next_state[i] = (
            0.5 * (game.n_firms - 1) * (1 - pi + p_other) * game.new_consumers
        )
    end

    return next_state
end
@everywhere SMPE.compute_action_bounds(
	game::SwitchingModel,
	state,
	player_ind,
	interp_value_function,
	actions
) = [(0., Inf)]
@everywhere SMPE.compute_optimization_x0(
    game::SwitchingModel, state, player_ind, interp_value_function, actions
) = isnothing(actions) ? [1.] : actions[player_ind]


# Correct equilibria for SwitchingModel. Computed from the source code as
# written by Einav & Somaini (available on soma.people.stanford.edu).
# s, df, N and g are the parameters of the model. beta, mu0, mu1 are the
# parametrization of the equilibrium policy function. We have that
# p_i = mean(c) + mu0 + mu1 (c_i - mean(c)) + beta * x_i.
struct SwitchingEq
	s
	df
	N
	g
	beta
	mu0
	mu1
end

switching_eqs = (
    SwitchingEq(0, 0.9, 2, 1, 0, 1, 0.3333),
    SwitchingEq(0.5, 0.9, 2, 1, 0.1681, 0.7940, 0.3656),
)
eq = switching_eqs[2]

mc = zeros(eq.N)
mc[1] = .5
# mc[0] = mc0

game = SwitchingModel(eq.N, mc, eq.s, eq.g, eq.df, 0)
grid = fill(range(0, 1, length=3), eq.N)
pf, vf = compute_equilibrium(game, grid)

# Einav and Somaini (2013, Theorem 1)
policy_check = map(i -> vec(collect(
	eq.mu0 + mean(mc) + eq.mu1 * (mc[i] - mean(mc)) + eq.beta * node[i]
	for node in Iterators.product(grid...)
)), 1:eq.N)

# We calculate the policies on a regular grid, so that in many of
# the initial states the market shares do not sum to 1. However, the
# policy functions from the paper are only valid when the market shares
# *do* sum to 1, so compare only for these nodes.
check_nodes = vec(collect(sum(node) == 1 for node in Iterators.product(grid...)))

@test [x[check_nodes] for x in pf] ≈ [x[check_nodes] for x in policy_check] atol=1e-2 rtol=1e-2
