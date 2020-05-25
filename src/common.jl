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
end

DEFAULT_OPTIONS = SMPEOptions()

"""
Wrapper type for interpolated functions. Stores the game for which the function
is interpolated with the function. The purpose of this is to support games
which are analyzed on non-rectangular grids. Implemen transform_state()
and related methods, to transform from a rectangular to non-rectangular grid.
The domain of the wrapper type is the transformeed (non-rectangular) state and
it takes care of the chain rule when calculating the graient or hessian.
"""
struct TransformedInterpolation{S<:AbstractInterpolation{N, T, NT} where {N, T, NT}, G<:DynamicGame}
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

struct Equilibrium{S <: AbstractArray{T, N} where {T <: Real, N}, T <: TransformedInterpolation}
    pf::Vector{Vector{T}}
    vf::Vector{T}
    attention::Matrix{Float64}
    pf_at_nodes::Vector{S}
    vf_at_nodes::S
    compute_time::Float64
end

perceived_state(
    game::DynamicGame,
    state::Vector{Float64},
    attention::Vector{Float64},
    default_state::Vector{Float64}
)::Vector{Float64} = attention .* state .+ (1 .- attention) .* default_state


function pad_actions(game::DynamicGame, actions_others::AbstractVector{<:AbstractVector{T}}, player_ind)::Vector{<:AbstractVector{T}} where T
    return vcat(
        actions_others[1:player_ind-1],
        [zeros(T, num_actions(game, player))],
        actions_others[player_ind:end]
    )
end
pad_actions(game::DynamicGame, actions_others::AbstractVector{<:AbstractArray}, player_ind)::Vector{Vector{Float64}} = [zeros(num_actions(game, player_ind))]

interpolate_value_function(game::DynamicGame, states, calc_value_function) = interpolate_function(game, states, calc_value_function)

interpolate_policy_function(game::DynamicGame, states, calc_policy_function) = [
    interpolate_function(game, states, calc_policy_function[fill(:, length(states))..., i])
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

eval_policy_function(policy_function, state) = map(
    pf -> pf(state...),
    policy_function
)

function value_function_for_state(
    game::DynamicGame,
    interp_value_function::TransformedInterpolation{<:AbstractInterpolation{T, N, NT}, <:DynamicGame},
    state::AbstractVector{<:Real},
    options::SMPEOptions
)::T where {T<:Real, N, NT}
    interp_value_function(state...)
end

function value_function_for_state(
    game::DynamicGame,
    interp_value_function::TransformedInterpolation{<:AbstractInterpolation{T, N, NT}, <:DynamicGame},
    state::AbstractVector{<:Union{<:Real, ContinuousUnivariateDistribution}},
    options::SMPEOptions
)::T where {T<:Real, N, NT}
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
    interp_value_function::TransformedInterpolation{<:AbstractInterpolation{T, N, NT}, <:DynamicGame},
    state::AbstractVector{<:Real},
    options::SMPEOptions
)::Vector{T} where {T<:Real, N, NT}
    Interpolations.gradient(interp_value_function, state...)
end

function value_function_gradient_for_state(
    game::DynamicGame,
    interp_value_function::TransformedInterpolation{<:AbstractInterpolation{T, N, NT}, <:DynamicGame},
    state::AbstractVector{<:Union{Real, ContinuousUnivariateDistribution}},
    options::SMPEOptions
)::Vector{T} where {T<:Real, N, NT}
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
    interp_value_function::TransformedInterpolation{<:AbstractInterpolation{T, N, NT}, <:DynamicGame},
    actions,
    actions_others,
    options::SMPEOptions
)::Vector{T} where {T<:Real, N, NT}
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
    Interpolations.gradient(interp_value_function, state...)
    return transpose(next_state_jac) * next_vf_grad
end

function value_function_gradient_for_actions(
    game::DynamicGame,
    state::AbstractVector{T},
    next_state::AbstractVector{<:Union{<:Real, ContinuousUnivariateDistribution}},
    player_ind,
    interp_value_function::TransformedInterpolation{<:AbstractInterpolation{T, N, NT}, <:DynamicGame},
    actions,
    actions_others,
    options::SMPEOptions
)::Vector{T} where {T<:Real, N, NT}
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
            transform_jac = transform_state_jac(game, integrand_node)
            cv_jac = prod(cv[2](y) for (cv, y) in zip(change_var, x))
            weight = prod(pdf(s, y) for (s, y) in zip(stochastic_nodes, x))
            f[:] = fn(integrand_state) * cv_jac * weight * abs(det(transform_jac))
        end,
        length(stochastic_nodes),
        dim_fn;
        rtol=options.integration_rel_tol,
        atol=options.integration_abs_tol
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
    interp_value_function::TransformedInterpolation{<:AbstractInterpolation{T, N, NT}, <:DynamicGame},
    states::AbstractVector{T},
    options::SMPEOptions
)::Matrix{T} where {T<:Real, N, NT}
    # There is no analytic hessian available for extrapolated functions
    # There is also a bug in Interpolations.jl for computing the Hssian when
    # ndim >= 4. See https://github.com/JuliaMath/Interpolations.jl/issues/364
    # So we just use ForwardDiff
    return ForwardDiff.hessian(x -> interp_value_function(x...), states)
end

function value_function_hessian_for_state(
    game::DynamicGame,
    interp_value_function::TransformedInterpolation{<:AbstractInterpolation{T, N, NT}, <:DynamicGame},
    state::AbstractVector{<:Union{Real, ContinuousUnivariateDistribution}},
    options::SMPEOptions
)::Matrix{T} where {T<:Real, N, NT}
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
