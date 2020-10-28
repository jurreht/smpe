@Base.kwdef struct SMPEOptions
    eps_outer::Real = 1e-4
    eps_inner::Real = 1e-4
    eps_contraction::Real = 1e-4
    optim_method::Optim.FirstOrderOptimizer = LBFGS()
    optim_options::Optim.Options = Optim.Options()
    state_sim_eps::Real = 1e-2
    state_sim_min_sims::Integer = 100
    integration_rel_tol::Real = 1e-1
    integration_abs_tol::Real = 1e-3
    integration_max_evals::Int64 = 1e6
    attention_cutoff::Real = 1e-4
    use_static_attention::Bool = false
    att_cost_bracket_step::Float64 = 5e-3
end

DEFAULT_OPTIONS = SMPEOptions()

abstract type SMPEInterpolation{T} end

"""
Wrapper type for interpolated functions. Stores the game for which the function
is interpolated with the function. The purpose of this is to support games
which are analyzed on non-rectangular grids. Implemen transform_state()
and related methods, to transform from a rectangular to non-rectangular grid.
The domain of the wrapper type is the transformeed (non-rectangular) state and
it takes care of the chain rule when calculating the graient or hessian.
"""
struct TransformedInterpolation{T, S <: AbstractInterpolation{T, N, IT} where {N, IT}, G <: DynamicGame} <: SMPEInterpolation{T}
    interpolation::S
    game::G
end

norm_vec(x::Tuple) = collect(Iterators.flatten(x))
norm_vec(x::AbstractArray) = x
(itp::TransformedInterpolation)(x...) = itp.interpolation(transform_state_back(itp.game, norm_vec(x))...)
Base.ndims(itp::TransformedInterpolation) = ndims(itp.interpolation)
Interpolations.gradient(itp::TransformedInterpolation, x...) = (
    transpose(transform_state_back_jac(itp.game, x)) *
    Interpolations.gradient(itp.interpolation, transform_state_back(itp.game, norm_vec(x))...)
)
function Interpolations.hessian(itp::TransformedInterpolation, x...)
    x = norm_vec(x)
    x_trans_back = transform_state_back(itp.game, x)
    grad_v = Interpolations.gradient(itp.interpolation, x_trans_back...)
    hess_v = Interpolations.hessian(itp.interpolation, x_trans_back...)
    jac_trans = transform_state_back_jac(itp.game, x)
    hess_trans = transform_state_back_hessian(itp.game, x)
    return (
        transpose(jac_trans) * hess_v * jac_trans +
        sum(hess_trans[i, :, :] .* grad_v[i] for i in 1:dim_rectangular_state(itp.game))
    )
end

struct SparseInterpolation{N, T, NT, S <: AbstractInterpolation{N, T, NT}} <: AbstractInterpolation{N, T, NT}
    itp::S
    active_states::BitArray{1}
    bounds::Vector{Tuple{Float64, Float64}}
end
# This is the type used in Interpolations.jl. Without using this type, method
# calls on SparseInterpolation are ambiguous
const VarArgType = Union{Number, AbstractArray{T, 1} where T, CartesianIndex}
#(itp::SparseInterpolation)(x::VarArgType...) = itp.itp(x[itp.active_states]...)
function (itp::SparseInterpolation)(x::VarArgType...)
    itp.itp(x[itp.active_states]...)
end
function Interpolations.gradient(itp::SparseInterpolation, x::VarArgType...)
    grad = zeros(length(itp.active_states))
    grad[itp.active_states] = Interpolations.gradient(itp.itp, x[itp.active_states]...)
    return grad
end
function Interpolations.hessian(itp::SparseInterpolation, x::VarArgType...)
    hess = zeros(length(itp.active_states), length(itp.active_states))
    hess[itp.active_states, itp.active_states] = Interpolations.hessian(itp.itp, x[itp.active_states]...)
    return hess
end

struct ConstantInterpolation{T <: Real} <: SMPEInterpolation{T}
    val::T
    bounds::Vector{Tuple{Float64, Float64}}
end
(itp::ConstantInterpolation)(x...) = itp.val
Interpolations.gradient(itp::ConstantInterpolation, x...) = zeros(length(x))
Interpolations.hessian(itp::ConstantInterpolation, x...) = zeros(length(x), length(x))

struct Equilibrium{T <: SMPEInterpolation}
    pf::Vector{Vector{T}}
    vf::Vector{T}
    attention::Matrix{Float64}
    compute_time::Float64
end

function perceived_state(
    game::DynamicGame,
    state::Vector{S},
    attention::Vector{T},
    default_state::Vector{Float64}
)::Vector{<:Real} where {S <: Real, T <: Real}
    attention .* state .+ (one(T) .- attention) .* default_state
end


function pad_actions(game::DynamicGame, actions_others::AbstractVector{<:AbstractVector{T}}, player_ind)::Vector{<:AbstractVector{T}} where T
    return vcat(
        actions_others[1:player_ind-1],
        [zeros(T, num_actions(game, player_ind))],
        actions_others[player_ind:end]
    )
end
pad_actions(game::DynamicGame, actions_others::AbstractVector{<:AbstractArray}, player_ind)::Vector{Vector{Float64}} = [zeros(num_actions(game, player_ind))]

interpolate_value_function(game::DynamicGame, states, calc_value_function) = interpolate_function(game, states, calc_value_function)
interpolate_value_function(game::DynamicGame, states, attention, calc_value_function, options::SMPEOptions=DEFAULT_OPTIONS) = interpolate_function(game, states, attention, calc_value_function, options)

interpolate_policy_function(game::DynamicGame, states, calc_policy_function) = [
    interpolate_function(game, states, calc_policy_function[fill(:, length(states))..., i])
    for i in 1:size(calc_policy_function, length(states) + 1)
]
interpolate_policy_function(game::DynamicGame, states, attention, calc_policy_function, options::SMPEOptions=DEFAULT_OPTIONS) = [
    interpolate_function(game, states, attention, calc_policy_function[fill(:, length(states))..., i], options)
    for i in 1:size(calc_policy_function, length(states) + 1)
]


function interpolate_function(game::DynamicGame, states, vals)
    itp = interpolate(vals, BSpline(Cubic(Line(OnGrid()))))
    # We allow extrapolation to prevent errors when the due to floating point
    # rounding we are just outside the grid. It is not advised to try to
    # extrapolate much beyond that
    etp = extrapolate(Interpolations.scale(itp, states...), Interpolations.Flat())
    return TransformedInterpolation(etp, game)
end

function interpolate_function(game::DynamicGame, states, attention, vals, options::SMPEOptions=DEFAULT_OPTIONS)
    bounds = [(minimum(s), maximum(s)) for s in states]
    notsparse = attention .> options.attention_cutoff
    if sum(notsparse) > 0
        if length(states) > sum(notsparse)
            states = states[notsparse]
        end
        if length(size(vals)) > sum(notsparse)
            vals = vals[(att ? Colon() : 1 for att in notsparse)...]
        end

        itp = interpolate(vals, BSpline(Cubic(Line(OnGrid()))))
        etp = extrapolate(Interpolations.scale(itp, states...), Interpolations.Flat())
        return TransformedInterpolation(SparseInterpolation(etp, notsparse, bounds), game)
    else
        # Full inattention, no point in interpolating
        return ConstantInterpolation(vals[1], bounds)
    end
end

eval_policy_function(policy_function, state) = map(
    pf -> pf(state...),
    policy_function
)

function value_function_for_state(
    game::DynamicGame,
    interp_value_function::SMPEInterpolation{T},
    state::AbstractVector{<:Real},
    options::SMPEOptions
)::T where {T<:Real}
    interp_value_function(state...)
end

function value_function_for_state(
    game::DynamicGame,
    interp_value_function::SMPEInterpolation{T},
    state::AbstractVector{<:Union{<:Real, ContinuousUnivariateDistribution}},
    options::SMPEOptions
)::T where {T<:Real}
    deterministic_states = map(s -> isa(s, T), state)
    node = transform_state_back(game, state)
    deterministic_nodes = map(s -> isa(s, T), node)
    return integrate_fn_over_stochastic_states(
        game,
        # We need an Array here since that is what is needed for numerical integration
        x -> [value_function_for_state(game, interp_value_function, x, options)],
        1,
        interpolation_bounds(interp_value_function),
        node,
        deterministic_nodes,
        options
    )[1]
end

function value_function_gradient_for_state(
    game::DynamicGame,
    interp_value_function::SMPEInterpolation{T},
    state::AbstractVector{<:Real},
    options::SMPEOptions
)::Vector{T} where {T<:Real, N, NT}
    Interpolations.gradient(interp_value_function, state...)
end

function value_function_gradient_for_state(
    game::DynamicGame,
    interp_value_function::SMPEInterpolation{T},
    state::AbstractVector{<:Union{Real, ContinuousUnivariateDistribution}},
    options::SMPEOptions
)::Vector{T} where {T<:Real}
    deterministic_states = map(s -> isa(s, T), state)
    node = transform_state_back(game, state)
    deterministic_nodes = map(s -> isa(s, T), node)
    return integrate_fn_over_stochastic_states(
        game,
        x -> value_function_gradient_for_state(game, interp_value_function, x, options)[deterministic_states],
        sum(deterministic_states),
        interpolation_bounds(interp_value_function),
        node,
        deterministic_nodes,
        options
    )
end

function value_function_gradient_for_actions(
    game::DynamicGame,
    state::AbstractVector{T},
    next_state::AbstractVector{<:Real},
    player_ind,
    interp_value_function::SMPEInterpolation{T},
    actions,
    actions_others,
    options::SMPEOptions
)::Vector{T} where {T<:Real}
    next_state_jac = ForwardDiff.jacobian(
        x -> compute_next_state(
            game,
            state,
            player_ind,
            x,
            pad_actions(game, actions_others, player_ind)
        ),
        actions
    )
    next_vf_grad = value_function_gradient_for_state(game, interp_value_function, next_state, options)
    return transpose(next_state_jac) * next_vf_grad
end

function value_function_gradient_for_actions(
    game::DynamicGame,
    state::AbstractVector{T},
    next_state::AbstractVector{<:Union{<:Real, ContinuousUnivariateDistribution}},
    player_ind,
    interp_value_function::SMPEInterpolation{T},
    actions,
    actions_others,
    options::SMPEOptions
)::Vector{T} where {T<:Real}
    next_node = transform_state_back(game, next_state)
    # Only differentiate wrt the non-stochastic parts of the state/node
    # (i.e. it is assumed that the stochastic state transitions are exogenous)
    deterministic_states = map(s -> isa(s, T), next_state)
    deterministic_nodes = map(s -> isa(s, T), next_node)
    # We can hence take the Jacobian of the state evolution wrt to
    # the deterministic states only
    next_state_jac = ForwardDiff.jacobian(
        x -> calculate_next_state_deterministic_part(
            deterministic_states,
            game,
            state,
            player_ind,
            x,
            pad_actions(game, actions_others, player_ind)
        ),
        actions
    )

    # Integrate out the random variables in the gradient of the value function
    # Integration needs [0, 1] box => use a change of variable
    next_vf_grad = integrate_fn_over_stochastic_states(
        game,
        x -> value_function_gradient_for_state(game, interp_value_function, x, options)[deterministic_states],
        sum(deterministic_states),
        interpolation_bounds(interp_value_function),
        next_node,
        deterministic_nodes,
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
    game::DynamicGame,
    fn,
    dim_fn,
    domain_fn,
    node,
    deterministic_nodes,
    options::SMPEOptions
)
    stochastic_nodes = node[.!deterministic_nodes]
    change_var = map(
        x -> make_scaler(x...),
        zip(stochastic_nodes, domain_fn[.!deterministic_nodes])
    )
    integrand_node = [
        is_deterministic ? x : NaN
        for (is_deterministic, x) in
        zip(deterministic_nodes, node)
    ]
    integral = vegas(
        (x, f) -> begin
            integrand_node[.!deterministic_nodes] = [
                cv[1](y) for (cv, y) in zip(change_var, x)
            ]
            integrand_state = transform_state(game, integrand_node)
            cv_jac = prod(cv[2](y) for (cv, y) in zip(change_var, x))
            weight = prod(pdf(s, y) for (s, y) in zip(stochastic_nodes, integrand_node[.!deterministic_nodes]))
            f[:] = fn(integrand_state) * cv_jac * weight
        end,
        length(stochastic_nodes),
        dim_fn;
        rtol=options.integration_rel_tol,
        atol=options.integration_abs_tol,
        maxevals=options.integration_max_evals
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
    game::DynamicGame,
    interp_value_function::SMPEInterpolation{T},
    states::AbstractVector{T},
    options::SMPEOptions
)::Matrix{T} where {T<:Real}
    # There is no analytic hessian available for extrapolated functions
    # There is also a bug in Interpolations.jl for computing the Hssian when
    # ndim >= 4. See https://github.com/JuliaMath/Interpolations.jl/issues/364
    # So we just use ForwardDiff
    return ForwardDiff.hessian(x -> interp_value_function(x...), states)
end

function value_function_hessian_for_state(
    game::DynamicGame,
    interp_value_function::SMPEInterpolation{T},
    state::AbstractVector{<:Union{Real, ContinuousUnivariateDistribution}},
    options::SMPEOptions
)::Matrix{T} where {T<:Real}
    node = transform_state_back(game, state)
    deterministic_states = map(s -> isa(s, T), state)
    deterministic_nodes = map(s -> isa(s, T), node)
    n_deterministic = sum(deterministic_states)
    hess_vectorized = integrate_fn_over_stochastic_states(
        game,
        # Integration only possible over vectors. Hence vectorize and reshape below
        x -> vec(value_function_hessian_for_state(game, interp_value_function, x, options)[deterministic_states, deterministic_states]),
        n_deterministic^2,
        interpolation_bounds(interp_value_function),
        node,
        deterministic_nodes,
        options
    )
    return reshape(hess_vectorized, n_deterministic, n_deterministic)
end

function interpolation_bounds(interp::TransformedInterpolation)
    itp = interp.interpolation

    if isa(itp, ConstantInterpolation) || isa(itp, SparseInterpolation)
        return itp.bounds
    end

    # Extrapolated functions have no bounds, give the bounds of the
    # underlying interpolated function
    vf_bounds = (isa(itp, AbstractExtrapolation)
        ? bounds(parent(itp))
        : bounds(itp)
    )
    return collect(zip(
        map(minimum, vf_bounds),
        map(maximum, vf_bounds)))
end

interpolation_bounds(interp::Union{ConstantInterpolation, SparseInterpolation}) = interp.bounds

function calculate_benefit_attention(
    game::DynamicGame,
    default_state,
    player_ind,
    interp_value_function,
    interp_policy_funcs,
    options::SMPEOptions,
    scale_by_state_variance=true
)
    actions_others = map(
        pf_player -> map(pf -> pf(default_state...), pf_player),
        interp_policy_funcs[1:end .!= player_ind]
    )
    if options.use_static_attention
        actions_own = eval_policy_function(interp_policy_funcs[player_ind], default_state)
        state_var = simulate_state_variance(game, default_state, player_ind, actions_own, interp_policy_funcs, options)
        padded_actions = pad_actions(game, actions_others, player_ind)
        x0 = collect(compute_optimization_x0(game, default_state, player_ind, nothing, nothing, padded_actions))
        bounds = compute_action_bounds(game, default_state, player_ind, nothing, nothing, padded_actions)
        lb = [isnothing(bounds[i][1]) ? -Inf : bounds[i][1] for i in eachindex(bounds)]
        ub = [isnothing(bounds[i][2]) ? Inf : bounds[i][2] for i in eachindex(bounds)]

        attention = Vector{Float64}(undef, dim_state(game))
        benefit_attention = Vector{Float64}(undef, dim_state(game))
        for state_ind in 1:dim_state(game)
            if any(-Inf .< lb) || any(Inf .> ub)
                res = Optim.optimize(
                    x -> -1 * static_benefit_attention(game, player_ind, default_state, x, actions_others, state_ind),
                    lb,
                    ub,
                    x0,
                    Fminbox(LBFGS());
                    autodiff=:forward
                )
            else
                res = Optim.optimize(
                    x -> -1 * static_benefit_attention(game, player_ind, default_state, x, actions_others, state_ind),
                    x0,
                    LBFGS();
                    autodiff=:forward
                )
            end
            if !Optim.converged(res)
                throw("No convergence when calculating attention")
            end
            benefit_attention[state_ind] = -.5 * state_var[state_ind] * Optim.minimum(res)
        end
        return benefit_attention
    else
        default_actions = calculate_optimal_actions(
            game,
            default_state,
            player_ind,
            interp_value_function,
            nothing,
            actions_others,
            options
        )
        n_actions_player = num_actions(game, player_ind)
        hess_value_actions = Matrix{Float64}(undef, n_actions_player, n_actions_player)
        # Calculate next_state here so that calculate_derivative_actions_state()
        # can dispath on its type
        next_state = compute_next_state(
            game,
            default_state,
            player_ind,
            default_actions,
            pad_actions(game, actions_others, player_ind)
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
        state_var = scale_by_state_variance ? simulate_state_variance(
                game,
                default_state,
                player_ind,
                default_actions,
                interp_policy_funcs,
                options
            ) : 1.
        return -.5 * state_var .* diag(
            transpose(deriv_actions_state) *
            hess_value_actions *
            deriv_actions_state
        )
    end
end

function calculate_relevant_nodes(
    game::DynamicGame, nodes, default_state, attention, options::SMPEOptions
)
    if all(attention .== 1)
        return Iterators.product(nodes...)
    else
        # Substitute default state for actual state if agent does not pay attention
        default_node = transform_state_back(game, default_state)
        return Iterators.product((
            attention[i] > options.attention_cutoff ? nodes[i] : [default_node[i]]
            for i in 1:dim_rectangular_state(game)
        )...)
    end
end

# getindex() is not implemented for either Enumerate or ProductIterator.
# But @threads() needs it to loop over the states. So we have it...
Base.firstindex(it::Base.Iterators.ProductIterator{<:Any}) = 1
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
