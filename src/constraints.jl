struct Constraint{T}
    evaluate
    jacobian_state
    jacobian_action
    hessian_prod_state_state
    hessian_prod_action_state
    hessian_prod_action_action
    num_constraint::Int
    num_state::Int
    num_action::Int
    num_parameter::Int
    evaluate_cache::Vector{T}
    jacobian_state_cache::Matrix{T}
    jacobian_action_cache::Matrix{T}
    hessian_prod_state_state_cache::Matrix{T}
    hessian_prod_action_state_cache::Matrix{T}
    hessian_prod_action_action_cache::Matrix{T}
    indices_inequality::Vector{Int}
    num_inequality::Int
end

Constraints{T} = Vector{Constraint{T}} where T

function Constraint(f::Function, num_state::Int, num_action::Int;
    indices_inequality::Vector{Int}=collect(1:0),
    num_parameter::Int=0, quasi_newton::Bool=false)

    #TODO: option to load/save methods
    x = Symbolics.variables(:x, 1:num_state)
    u = Symbolics.variables(:u, 1:num_action)
    w = Symbolics.variables(:w, 1:num_parameter)
    # @variables x[1:num_state], u[1:num_action], w[1:num_parameter]

    evaluate = num_parameter > 0 ? f(x, u, w) : f(x, u)
    jacobian_state = Symbolics.jacobian(evaluate, x)
    jacobian_action = Symbolics.jacobian(evaluate, u)

    evaluate_func = eval(Symbolics.build_function(evaluate, x, u, w)[2])
    jacobian_state_func = eval(Symbolics.build_function(jacobian_state, x, u, w)[2])
    jacobian_action_func = eval(Symbolics.build_function(jacobian_action, x, u, w)[2])

    num_constraint = length(evaluate)
    num_inequality = length(indices_inequality) # Add this for bookkeeping of number of inequalities
    
    v = Symbolics.variables(:v, 1:num_inequality)  # vector variables for Hessian vector products
    
    if !quasi_newton
        hessian_prod_state_state = sum([v[k] .* Symbolics.hessian(evaluate[k], x) for k in 1:num_constraint])
        hessian_prod_action_state = sum([v[k] .* Symbolics.jacobian(Symbolics.gradient(evaluate[k], u), x) for k in 1:num_constraint])
        hessian_prod_action_action = sum([v[k] .* Symbolics.hessian(evaluate[k], u) for k in 1:num_constraint])
        hessian_prod_state_state_func = eval(Symbolics.build_function(hessian_prod_state_state, x, u, w, v)[2])
        hessian_prod_action_state_func = eval(Symbolics.build_function(hessian_prod_action_state, x, u, w, v)[2])
        hessian_prod_action_action_func = eval(Symbolics.build_function(hessian_prod_action_action, x, u, w, v)[2])
    else
        hessian_prod_state_state_func = nothing
        hessian_prod_action_state_func = nothing
        hessian_prod_action_action_func = nothing
    end

    return Constraint(
        evaluate_func,
        jacobian_state_func, jacobian_action_func, hessian_prod_state_state_func, hessian_prod_action_state_func,
        hessian_prod_action_action_func,
        num_constraint, num_state, num_action, num_parameter,
        zeros(num_constraint), zeros(num_constraint, num_state), zeros(num_constraint, num_action),
        zeros(num_state, num_state), zeros(num_action, num_state), zeros(num_action, num_action),
        indices_inequality, num_inequality)
end

function Constraint()
    return Constraint(
        (c, x, u, w) -> nothing,
        (jx, x, u, w) -> nothing, (ju, x, u, w) -> nothing,
        (hxx, x, u, w, v) -> nothing, (hux, x, u, w, v) -> nothing, (huu, x, u, w, v) -> nothing,
        0, 0, 0, 0,
        Float64[], Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0),
        Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0),
        collect(1:0), 0)
end

function Constraint(f::Function, fx::Function, fu::Function, num_constraint::Int, num_state::Int, num_action::Int;
    indices_inequality::Vector{Int}=collect(1:0),
    num_parameter::Int=0, fxx_prod::Function=nothing, fux_prod::Function=nothing, fuu_prod::Function=nothing)

    return Constraint(
        f,
        fx, fu, fxx_prod, fux_prod, fuu_prod,
        num_constraint, num_state, num_action, num_parameter,
        zeros(num_constraint), zeros(num_constraint, num_state), zeros(num_constraint, num_action),
        zeros(num_state, num_state), zeros(num_action, num_state), zeros(num_action, num_action),
        indices_inequality, length(indices_inequality))
end

function jacobian!(jacobian_states, jacobian_actions, constraints::Constraints{T}, states, actions, parameters) where T
    N = length(constraints)
    for (k, con) in enumerate(constraints)
        con.num_constraint == 0 && continue
        con.jacobian_state(con.jacobian_state_cache, states[k], actions[k], parameters[k])
        @views jacobian_states[k] .= con.jacobian_state_cache
        fill!(con.jacobian_state_cache, 0.0) # TODO: confirm this is necessary
        k == N && continue
        con.jacobian_action(con.jacobian_action_cache, states[k], actions[k], parameters[k])
        @views jacobian_actions[k] .= con.jacobian_action_cache
        fill!(con.jacobian_action_cache, 0.0) # TODO: confirm this is necessary
    end
end

function hessian_vector_prod!(hessian_prod_state_state, hessian_prod_action_state, hessian_prod_action_action,
    constraint::Constraint{T}, states, actions, parameters, lhs_vector) where T
    constraint.hessian_prod_state_state(constraint.hessian_prod_state_state_cache, states, actions, parameters, lhs_vector)
    constraint.hessian_prod_action_state(constraint.hessian_prod_action_state_cache, states, actions, parameters, lhs_vector)
    constraint.hessian_prod_action_action(constraint.hessian_prod_action_action_cache, states, actions, parameters, lhs_vector)
    @views hessian_prod_state_state .= constraint.hessian_prod_state_state_cache
    @views hessian_prod_action_state .= constraint.hessian_prod_action_state_cache
    @views hessian_prod_action_action .= constraint.hessian_prod_action_action_cache
end
