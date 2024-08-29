struct Constraint{T}
    evaluate
    jacobian_state
    jacobian_action
    hessian_prod_state_state
    hessian_prod_action_state
    hessian_prod_action_action
    num_constraint::Int
    num_ineq_lower::Int
    num_ineq_upper::Int
    num_state::Int
    num_action::Int
    evaluate_cache::Vector{T}
    jacobian_state_cache::Matrix{T}
    jacobian_action_cache::Matrix{T}
    hessian_prod_state_state_cache::Matrix{T}
    hessian_prod_action_state_cache::Matrix{T}
    hessian_prod_action_action_cache::Matrix{T}
    indices_compl::Vector{Int}
end

Constraints{T} = Vector{Constraint{T}} where T

function Constraint(f::Function, num_state::Int, num_action::Int; bounds_lower::Vector{T}=ones(num_action) * -Inf,
    bounds_upper::Vector{T}=ones(num_action) * Inf, quasi_newton::Bool=false, indices_compl=nothing) where T

    x = Symbolics.variables(:x, 1:num_state)
    u = Symbolics.variables(:u, 1:num_action)

    evaluate = f(x, u)
    jacobian_state = Symbolics.jacobian(evaluate, x)
    jacobian_action = Symbolics.jacobian(evaluate, u)

    evaluate_func = eval(Symbolics.build_function(evaluate, x, u)[2])
    jacobian_state_func = eval(Symbolics.build_function(jacobian_state, x, u)[2])
    jacobian_action_func = eval(Symbolics.build_function(jacobian_action, x, u)[2])

    num_constraint = length(evaluate)
    num_ineq_lower = sum(isfinite, bounds_lower)
    num_ineq_upper = sum(isfinite, bounds_upper)

    if length(bounds_lower) != num_action || length(bounds_upper) != num_action
        error("Bounds provided do not match dimension of actions.")
    end
    
    v = Symbolics.variables(:v, 1:num_constraint)  # vector variables for Hessian vector products

    if !quasi_newton && num_constraint > 0
        hessian_prod_state_state = sum([Symbolics.jacobian(v[k] .* jacobian_state[k, :], x) for k in 1:num_constraint])
        hessian_prod_action_state = sum([Symbolics.jacobian(v[k] .* jacobian_action[k, :], x) for k in 1:num_constraint])
        hessian_prod_action_action = sum([Symbolics.jacobian(v[k] .* jacobian_action[k, :], u) for k in 1:num_constraint])
        hessian_prod_state_state_func = eval(Symbolics.build_function(hessian_prod_state_state, x, u, v)[2])
        hessian_prod_action_state_func = eval(Symbolics.build_function(hessian_prod_action_state, x, u, v)[2])
        hessian_prod_action_action_func = eval(Symbolics.build_function(hessian_prod_action_action, x, u, v)[2])
    else
        hessian_prod_state_state_func = nothing
        hessian_prod_action_state_func = nothing
        hessian_prod_action_action_func = nothing
    end

    indices_compl = isnothing(indices_compl) ? Int64[] : indices_compl 

    return Constraint(
        evaluate_func,
        jacobian_state_func, jacobian_action_func, 
        hessian_prod_state_state_func, hessian_prod_action_state_func, hessian_prod_action_action_func,
        num_constraint, num_ineq_lower, num_ineq_upper, num_state, num_action,
        zeros(num_constraint), zeros(num_constraint, num_state), zeros(num_constraint, num_action),
        zeros(num_state, num_state), zeros(num_action, num_state), zeros(num_action, num_action),
        indices_compl)
end

function Constraint()
    return Constraint(
        (c, x, u) -> nothing,
        (jx, x, u) -> nothing, (ju, x, u) -> nothing,
        (hxx, x, u, v) -> nothing, (hux, x, u, v) -> nothing, (huu, x, u, v) -> nothing,
        0, 0, 0, 0, 0,
        Float64[], Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0),
        Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0),
        Int64[])
end

function Constraint(f::Function, fx::Function, fu::Function, num_constraint::Int, num_state::Int, num_action::Int;
    bounds_lower::Vector{T}=ones(num_action) *-Inf, bounds_upper::Vector{T}=ones(num_action) * Inf,
    indices_compl=nothing,
    fxx_prod::Function=nothing, fux_prod::Function=nothing, fuu_prod::Function=nothing) where T

    num_ineq_lower = sum(isfinite, bounds_lower)
    num_ineq_upper = sum(isfinite, bounds_upper)

    return Constraint(
        f, 
        fx, fu,
        fxx_prod, fux_prod, fuu_prod,
        num_constraint, num_state, num_action, num_ineq_lower, num_ineq_upper,
        zeros(num_constraint), zeros(num_constraint, num_state), zeros(num_constraint, num_action),
        zeros(num_state, num_state), zeros(num_action, num_state), zeros(num_action, num_action),
        indices_compl)
end

function jacobian!(jacobian_states, jacobian_actions, constraints::Constraints{T}, states, actions) where T
    N = length(constraints)
    for (k, con) in enumerate(constraints)
        con.num_constraint == 0 && continue
        con.jacobian_state(jacobian_states[k], states[k], actions[k])
        con.jacobian_action(jacobian_actions[k], states[k], actions[k])
    end
end

function hessian_vector_prod!(hessian_prod_state_state, hessian_prod_action_state, hessian_prod_action_action,
    constraint::Union{Dynamics{T}, Constraint{T}}, states, actions, lhs_vector) where T
    if !isnothing(constraint.hessian_prod_state_state)
        constraint.hessian_prod_state_state(hessian_prod_state_state, states, actions, lhs_vector)
        constraint.hessian_prod_action_state(hessian_prod_action_state, states, actions, lhs_vector)
        constraint.hessian_prod_action_action(hessian_prod_action_action, states, actions, lhs_vector)
    end
end
