using Distributions
using Logging
using Test
using Random
using SMPE
using Statistics

disable_logging(Logging.Info)

# Test games
"""
A standard, one player, capital accumulation problem. This has
closed form solutions, therefore it is easy to verify our numerical
results. This is based on Example 6.4 from Acemoglu, "An Introduction
to Modern Economic growth."
"""
struct CapitalAccumulationProblem{T<:Real} <: DynamicGame
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

SMPE.num_players(g::CapitalAccumulationProblem) = 1
SMPE.dim_state(g::CapitalAccumulationProblem) = 1
SMPE.num_actions(g::CapitalAccumulationProblem, i) = 1
SMPE.discount_factor(g::CapitalAccumulationProblem, i) = g.beta
SMPE.static_payoff(
	game::CapitalAccumulationProblem,
	state,
	player_ind,
	actions_player,
	actions
	) = log(state[1] ^ game.alpha - actions_player[1])
SMPE.compute_next_state(
	game::CapitalAccumulationProblem,
	state,
	player_ind,
	actions_player,
	actions
) = actions_player
SMPE.compute_action_bounds(
	game::CapitalAccumulationProblem,
	state,
	player_ind,
	interp_value_function,
	actions_player,
	actions
) = [(game.k_lowest, state[1] ^ game.alpha)]
SMPE.compute_optimization_x0(
	game::CapitalAccumulationProblem,
	state,
	player_ind,
	interp_value_function,
	actions_player,
	actions
	) = isnothing(actions_player) ? [state[1] ^ game.alpha / 2] : actions_player
SMPE.attention_cost(game::CapitalAccumulationProblem, player_ind) = 0

abstract type SwitchingModel <: DynamicGame end

"""
A model of competition with switching costs (Einav and Somaini, 2013).

Somaini, Paulo and Einav, Liran. 2013. "A Model of Market Power in
Customer Markets." Journal of Industral Economics 61 (4): 938-986.
"""
struct DeterministicSwitchingModel <: SwitchingModel
	n_firms::Int
	marginal_cost::Vector{Float64}
	switch_cost::Float64
    new_consumers::Float64
    df_firms::Float64
    df_consumers::Float64
	att_cost::Float64
end

SMPE.num_players(game::SwitchingModel) = game.n_firms
SMPE.num_actions(game::SwitchingModel, i) = 1
SMPE.discount_factor(game::SwitchingModel, i) = game.df_firms
SMPE.attention_cost(game::SwitchingModel, player_ind) = game.att_cost
SMPE.compute_action_bounds(
	game::SwitchingModel,
	state,
	player_ind,
	interp_value_function,
	actions_player,
	actions
) = [(0., Inf)]
SMPE.compute_optimization_x0(
    game::SwitchingModel, state, player_ind, interp_value_function, actions_player, actions
) = isnothing(actions_player) ? [1.] : actions_player

SMPE.dim_state(game::DeterministicSwitchingModel) = game.n_firms - 1
SMPE.dim_rectangular_state(game::DeterministicSwitchingModel) = game.n_firms - 1
function SMPE.static_payoff(
	game::DeterministicSwitchingModel,
	state,
	player_ind,
	actions_player,
	actions
)
	pi = actions_player[1]
	p_other = mean(x[1] for x in actions[1:end .!= player_ind])
	xi = player_ind <= SMPE.dim_state(game) ? state[player_ind] : 1 - sum(state)
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
function SMPE.compute_next_state(
	game::DeterministicSwitchingModel,
	state,
	player_ind,
	actions_player::AbstractVector{T},
	actions
) where T <: Real
	# When there is automatic differentiation, actions[player_ind] may be some
	# kind of DualNumber. Then zeros() need to create an array of the same type
	# otherwise we will not be able to assign to next_state below.
	next_state = zeros(T, game.n_firms - 1)

	prices = [i == player_ind ? actions_player[1] : x[1] for (i, x) in enumerate(actions)]

	market_size = game.n_firms * (game.n_firms - 1) / 2
	for i in 1:(game.n_firms - 1)
		pi = prices[i]
		p_other = mean(prices[1:game.n_firms .!= i])

		# Marhet share = demand young / market size
		# Demand young is given by eq 12 in Somaini & Einav (2013), noting
		# that we only consider the case where consumers are myopic.
		# Market size is "M" in Somaini & Einav (2013).
		next_state[i] = (
			0.5 * (game.n_firms - 1) * (1 - pi + p_other) * game.new_consumers
		) / market_size
	end

	return next_state
end
function SMPE.compute_default_state(game::DeterministicSwitchingModel, player_ind)
	n_players = SMPE.num_players(game)
	return fill(1 / n_players, n_players - 1)
end
SMPE.transform_state(::DeterministicSwitchingModel, state) = SMPE.rectangular_to_simplex(state)
SMPE.transform_state_jac(::DeterministicSwitchingModel, state) = SMPE.rectangular_to_simplex_jac(state)
SMPE.transform_state_back(::DeterministicSwitchingModel, state) = SMPE.simplex_to_rectangular(state)
SMPE.transform_state_back_jac(::DeterministicSwitchingModel, state) = SMPE.simplex_to_rectangular_jac(state)
SMPE.transform_state_back_hessian(::DeterministicSwitchingModel, state) = SMPE.simplex_to_rectangular_hess(state)

struct StochasticSwitchingModel{T <: Real} <: SwitchingModel
	n_firms::Int
	marginal_cost::Vector{T}
	switch_cost::Float64
	mean_g::Float64
	corr_g::Float64
	df_firms::Float64
	df_consumers::Float64
	att_cost::Float64
end
SMPE.dim_rectangular_state(game::StochasticSwitchingModel) = game.n_firms + 1
SMPE.dim_state(game::StochasticSwitchingModel) = game.n_firms + 1
SMPE.transform_state(::StochasticSwitchingModel, state) = vcat(
	SMPE.rectangular_to_simplex(state[1:end-2]),
	collect(state[end-1:end])
)
function SMPE.transform_state_jac(game::StochasticSwitchingModel, state)
	ret = zeros(game.n_firms + 1, game.n_firms + 1)
	ret[1:game.n_firms - 1, 1:game.n_firms - 1] = SMPE.rectangular_to_simplex_jac(state[1:end-2])
	ret[game.n_firms, game.n_firms] = 1
	ret[game.n_firms + 1, game.n_firms + 1] = 1
	return ret
end
SMPE.transform_state_back(::StochasticSwitchingModel, state) = vcat(
	SMPE.simplex_to_rectangular(state[1:end-2]),
	collect(state[end-1:end])
)
function SMPE.transform_state_back_jac(game::StochasticSwitchingModel, state)
	ret = zeros(game.n_firms + 1, game.n_firms + 1)
	ret[1:game.n_firms - 1, 1:game.n_firms - 1] = SMPE.simplex_to_rectangular_jac(state[1:end-2])
	ret[game.n_firms, game.n_firms] = 1
	ret[game.n_firms + 1, game.n_firms + 1] = 1
	return ret
end
function SMPE.transform_state_back_hessian(game::StochasticSwitchingModel, state)
	ret = zeros(game.n_firms + 1, game.n_firms + 1, game.n_firms + 1)
	ret[1:game.n_firms - 1, 1:game.n_firms - 1, 1:game.n_firms - 1] = SMPE.simplex_to_rectangular_hess(state)
	return ret
end
function SMPE.static_payoff(
	game::StochasticSwitchingModel,
	state,
	player_ind,
	actions_player,
	actions
)
	pi = actions_player[1]
	p_other = mean(x[1] for x in actions[1:end .!= player_ind])
	xi = player_ind <= game.n_firms - 1 ? state[player_ind] : 1 - sum(state[1:game.n_firms - 1])
	f = state[game.n_firms]
	g = state[game.n_firms + 1]
	ci = game.marginal_cost[player_ind]

	# eq. 10
	demand_old = (
		0.5
		* (game.n_firms - 1)
		* ((1 - pi + p_other) + (game.n_firms * xi - 1) * game.switch_cost)
	) * f

	# eq. 12
	demand_young = (
		0.5 * (game.n_firms - 1) * (1 - pi + p_other) * g
	) * f

	# first part of eq. 22
	return (pi - ci) * (demand_young + demand_old)
end
function SMPE.compute_next_state(
	game::StochasticSwitchingModel,
	state,
	player_ind,
	actions_player,
	actions
)
	if isnothing(actions_player)
		return [(0., Inf)]
	end

	next_state = []
	prices = [i == player_ind ? actions_player[1] : x[1] for (i, x) in enumerate(actions)]

	f = state[game.n_firms]
	g = state[game.n_firms + 1]
	# Ignore f * g as it would get divided out below anyay
	market_size = game.n_firms * (game.n_firms - 1) / 2
	for i in 1:(game.n_firms - 1)
		pi = prices[i]
		p_other = mean(prices[1:game.n_firms .!= i])

		push!(
			next_state,
			0.5 * (game.n_firms - 1) * (1 - pi + p_other) / market_size
		)
	end

	push!(next_state, f * g)
	push!(next_state, Distributions.truncated(Normal(
		(1 - game.corr_g) * game.mean_g + game.corr_g * g,
		.2
	), 0, 1 / game.df_firms))  # upper lim need to get discounting

	return convert(Vector{Union{<:Real, ContinuousUnivariateDistribution}}, next_state)
end
function SMPE.compute_default_state(game::StochasticSwitchingModel, player_ind)
	n_players = SMPE.num_players(game)
	return vcat(fill(1 / n_players, n_players - 1), 1., game.mean_g)
	# return vcat(fill(0, n_players), game.mean_L, game.mean_L)
end

# Static NE of Somaini & Einav (2013) given by eq 5
somaini_einav_static_eq(game, grid) = map(mci -> fill(
	1 + mean(game.marginal_cost) + (game.n_firms - 1) / (2 * game.n_firms - 1) * (mci - mean(game.marginal_cost)),
	size(Iterators.product(grid...))
), game.marginal_cost)

@time @testset "Compute and estimation tests" begin
	@testset "Capital accumulation problem" begin
		alpha = 0.5
		beta = 0.95
		game = CapitalAccumulationProblem(alpha, beta)
		grid = range(game.k_lowest, 4*game.k_steady_state, length=100)
		eq = compute_equilibrium(game, grid)
		policy_exact = alpha * beta * collect(grid) .^ alpha
		@test (@. eq.pf[1][1](grid)) ≈ policy_exact atol=1e-2 rtol=5e-1
		@test all(eq.attention .== 1)
	end

	# Correct equilibria for SwitchingModel. Computed from the source code as
	# written by Somaini & Einav (available on soma.people.stanford.edu).
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
	# The last three coefficients are the coefficients of the equilibrium
	# policy functions. They are calculated using the code from Somaini's
	# website.
	switching_eqs = (
	    SwitchingEq(0., 0.9, 2, 1., 0, 1, 0.3333),
	    SwitchingEq(0.5, 0.9, 2, 1., 0.1681, 0.7940, 0.3656),
		SwitchingEq(0., 0.9, 3, 1., 0, 1, 0.4),
		SwitchingEq(0.5, 0.9, 3, 1., 0.3033, 0.7524, 0.4429),
	)

	@testset "Somaini and Einav (2013) without attention costs" for (eq, mc1) in Iterators.product(switching_eqs, (0, .5))
		mc = zeros(eq.N)
		mc[1] = mc1

		game = DeterministicSwitchingModel(eq.N, mc, eq.s, eq.g, eq.df, 0, 0)
		grid = fill(range(0.01, .99, length=10), eq.N - 1)
		out = compute_equilibrium(game, grid)

		# Somaini and Einav (2013, Theorem 1)
		for i in 1:eq.N
			policy_check = map(
				state -> eq.mu0 + mean(mc) + eq.mu1 * (mc[i] - mean(mc)) + eq.beta * (i < game.n_firms ? state[i] : 1 - sum(state)),
				map(node -> SMPE.transform_state(game, node), Iterators.product(grid...)),
			)
			@test map(node -> out.pf[i][1](node), Iterators.product(grid...)) ≈ policy_check atol=1e-2 rtol=1e-2
		end
		@test all(out.attention .== 1)
	end

	@testset "Somaini and Einav (2013) with attention costs" for (eq, mc1) in Iterators.product(switching_eqs, [0])#, .5])
		mc = zeros(eq.N)
		mc[1] = mc1

		game = DeterministicSwitchingModel(eq.N, mc, eq.s, eq.g, eq.df, 0, .1)
		grid = fill(range(0.01, .99, length=10), eq.N - 1)
		out = compute_equilibrium(game, grid)

		# The SMPE is just no attention to anything and repeated play
	    # of the static Nash equilibrium for the steady state market
	    # shares.
		policy_check = somaini_einav_static_eq(game, grid)

		for i in 1:eq.N
			@test map(node -> out.pf[i][1](node), Iterators.product(grid...)) ≈ policy_check[i] atol=1e-2 rtol=1e-2
		end
		@test all(out.attention .== 0.)
	end

	@testset "Somain and Einav (2013) with stochastic market size" for (eq, mc1) in Iterators.product(
		switching_eqs[[eq.s == 0 for eq in switching_eqs]], [0, .5]
	)
		mc = zeros(eq.N)
		mc[1] = mc1

		game = StochasticSwitchingModel(eq.N, mc, eq.s, eq.g, .7, eq.df, 0., .1)
		grid = vcat(
			fill(range(.01, .99, length=10), eq.N - 1),
			[range(.1, 2, length=10)],
			[range(0, 1 / game.df_firms, length=10)]
		)

		out = compute_equilibrium(game, grid)

		# Without switching costs, equilibrium is repetition of static equilibrium
		policy_check = somaini_einav_static_eq(game, grid)

		for i in 1:eq.N
			@test map(node -> out.pf[i][1](node), Iterators.product(grid...)) ≈ policy_check[i] atol=1e-2 rtol=1e-2
		end
		@test all(out.attention .== 0)

		# Test if equilibrium is the same if we fix attention
		out_fixed_att = compute_equilibrium(game, grid; fix_attention=out.attention)
		for i in 1:eq.N
			@test map(node -> out.pf[i][1](node), Iterators.product(grid...)) == map(node -> out.pf[i][1](node), Iterators.product(grid...))
		end

		# Now estimate the model
		Random.seed!(452)
		path = simulate_equilibrium_path(game, out, SMPE.compute_default_state(game, 1), 100)
		for player_ind in 1:eq.N
			kappa_bounds = estimate_attention_cost(game, player_ind, out.attention[player_ind, :], path, out.pf, grid, (0., .5), .05)
			@test kappa_bounds == (0., .5)
		end
	end
end
