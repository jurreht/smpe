using Distributed

if nprocs() < 4
	addprocs(4; exeflags="--project")
end

@everywhere using Distributions
using Logging
using Test
@everywhere using SMPE
@everywhere using Statistics

disable_logging(Logging.Info)

# Test games
@everywhere begin
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
		actions
		) = log(state[1] ^ game.alpha - actions[1][1])
	SMPE.compute_next_state(
		game::CapitalAccumulationProblem,
		state,
		player_ind,
		actions
	) = actions[1]
	SMPE.compute_action_bounds(
		game::CapitalAccumulationProblem,
		state,
		player_ind,
		interp_value_function,
		actions
	) = [(game.k_lowest, state[1] ^ game.alpha)]
	SMPE.compute_optimization_x0(
		game::CapitalAccumulationProblem,
		state,
		player_ind,
		interp_value_function,
		actions
		) = isnothing(actions) ? [state[1] ^ game.alpha / 2] : actions[1]
	SMPE.attention_cost(game::CapitalAccumulationProblem, player_ind) = 0

	abstract type SwitchingModel <: DynamicGame end

	"""
	A model of competition with switching costs (Einav and Somaini, 2013).

	Somaini, Paulo and Einav, Liran. 2013. "A Model of Market Power in
	Customer Markets." Journal of Industral Economics 61 (4): 938-986.
	"""
	struct DeterministicSwitchingModel <: SwitchingModel
		n_firms
		marginal_cost
		switch_cost
	    new_consumers
	    df_firms
	    df_consumers
		att_cost
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
		actions
	) = [(0., Inf)]
	SMPE.compute_optimization_x0(
	    game::SwitchingModel, state, player_ind, interp_value_function, actions
	) = isnothing(actions) ? [1.] : actions[player_ind]

	SMPE.dim_state(game::DeterministicSwitchingModel) = game.n_firms
	function SMPE.static_payoff(
		game::DeterministicSwitchingModel,
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
	function SMPE.compute_next_state(
		game::DeterministicSwitchingModel,
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
	function SMPE.compute_default_state(game::DeterministicSwitchingModel, player_ind)
		n_players = SMPE.num_players(game)
		return fill(1 / n_players, n_players)
	end

	struct StochasticSwitchingModel <: SwitchingModel
		n_firms
		marginal_cost
		switch_cost
		mean_L
		corr_L
		df_firms
		df_consumers
		att_cost
	end
	SMPE.dim_state(game::StochasticSwitchingModel) = game.n_firms + 2
	function SMPE.static_payoff(
		game::StochasticSwitchingModel,
		state,
		player_ind,
		actions
	)
		pi = actions[player_ind][1]
		p_other = mean(x[1] for x in actions[1:end .!= player_ind])
		xi = state[player_ind]
		L = state[game.n_firms + 1]
		L_past = state[game.n_firms + 1]
		g = (L - L_past) / (1 + L - L_past)
		ci = game.marginal_cost[player_ind]

		# eq. 10
		demand_old = (
			0.5
			* (game.n_firms - 1)
			* ((1 - pi + p_other) + (game.n_firms * xi - 1) * game.switch_cost)
		)

		# eq. 12
		demand_young = (
			0.5 * (game.n_firms - 1) * (1 - pi + p_other) * g
		)

		# first part of eq. 22
		return L * (pi - ci) * (demand_young + demand_old)
	end
	function SMPE.compute_next_state(
		game::StochasticSwitchingModel,
		state,
		player_ind,
		actions,
	)
		next_state = []

		L = state[game.n_firms + 1]
		L_past = state[game.n_firms + 1]
		g = (L - L_past) / (1 + L - L_past)
		for i in 1:game.n_firms
			pi = actions[i][1]
			p_other = mean(x[1] for x in actions[1:game.n_firms .!= i])

			push!(
				next_state,
				0.5 * (game.n_firms - 1) * (1 - pi + p_other) * g
			)
		end

		push!(next_state, LogNormal(
			(1 - game.corr_L) * log(game.mean_L) - .02 + game.corr_L * log(L),
			0.2
		))
		push!(next_state, L)

		state_type = typeof(actions[player_ind]).parameters[1]
		return convert(Vector{Union{<:Real, LogNormal}}, next_state)
	end
	function SMPE.compute_default_state(game::StochasticSwitchingModel, player_ind)
		n_players = SMPE.num_players(game)
		return vcat(fill(1 / n_players, n_players), game.mean_L, game.mean_L)
	end
end

# Static NE of Somaini & Einav (2013) given by eq 5
somaini_einav_static_eq(mc, grid) = map(mci -> fill(
	1 + mean(mc) + .333 * (mci - mean(mc)),
	size(Iterators.product(grid...))
), mc)

@time @testset "Compute tests" begin
	@testset "Capital accumulation problem" begin
		alpha = 0.5
		beta = 0.95
		game = CapitalAccumulationProblem(alpha, beta)
		grid = range(game.k_lowest, 4*game.k_steady_state, length=100)
		pf, vf, att = compute_equilibrium(game, grid; return_interpolated=false)
		policy_exact = alpha * beta * collect(grid) .^ alpha
		@test pf[1] ≈ policy_exact atol=1e-2 rtol=5e-1
		@test all(att)
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
	switching_eqs = (
	    SwitchingEq(0, 0.9, 2, 1, 0, 1, 0.3333),
	    SwitchingEq(0.5, 0.9, 2, 1, 0.1681, 0.7940, 0.3656),
	)

	@testset "Somaini and Einav (2013) without attention costs" for (eq, mc1) in Iterators.product(switching_eqs, (0, .5))
		mc = zeros(eq.N)
		mc[1] = mc1

		game = DeterministicSwitchingModel(eq.N, mc, eq.s, eq.g, eq.df, 0, 0)
		grid = fill(range(0, 1, length=10), eq.N)
		pf, vf, att = compute_equilibrium(game, grid; return_interpolated=false)

		# Somaini and Einav (2013, Theorem 1)
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
		@test all(att)
	end

	@testset "Somaini and Einav (2013) with attention costs" for (eq, mc1) in Iterators.product(switching_eqs, [0, .5])
		mc = zeros(eq.N)
		mc[1] = mc1

		game = DeterministicSwitchingModel(eq.N, mc, eq.s, eq.g, eq.df, 0, .1)
		grid = fill(range(0, 1, length=10), eq.N)
		pf, vf, att = compute_equilibrium(game, grid; return_interpolated=false)

		# The SMPE is just no attention to anything and repeated play
	    # of the static Nash equilibrium for the steady state market
	    # shares.
		policy_check = somaini_einav_static_eq(mc, grid)

		# We calculate the policies on a regular grid, so that in many of
		# the initial states the market shares do not sum to 1. However, the
		# policy functions from the paper are only valid when the market shares
		# *do* sum to 1, so compare only for these nodes.
		check_nodes = vec(collect(sum(node) == 1 for node in Iterators.product(grid...)))

		@test [x[check_nodes] for x in pf] ≈ [x[check_nodes] for x in policy_check] atol=1e-2 rtol=1e-2
		@test !any(att)
	end

	@testset "Somain and Einav (2013) with stochastic market size" for (eq, mc1) in Iterators.product(
		switching_eqs[[eq.s == 0 for eq in switching_eqs]], [0, .5]
	)
		mc = zeros(eq.N)
		mc[1] = mc1

		game = StochasticSwitchingModel(eq.N, mc, eq.s, eq.g, .7, eq.df, 0, .1)
		grid = [
			range(0, 1, length=10),
			range(0, 1, length=10),
			range(.1, 2, length=10),
			range(.1, 2, length=10)
		]

		pf, vf, att = compute_equilibrium(game, grid; return_interpolated=false)

		# Without switching costs, equilibrium is repetition of static equilibrium
		policy_check = somaini_einav_static_eq(mc, grid)

		check_nodes = vec(collect(sum(node) == 1 for node in Iterators.product(grid...)))

		@test [x[check_nodes] for x in pf] ≈ [x[check_nodes] for x in policy_check] atol=1e-2 rtol=1e-2
		@test !any(att)
	end
end
