struct Dynamics{T}
    evaluate
    jacobian_state
    jacobian_action
    hessian_prod_state_state
    hessian_prod_action_state
    hessian_prod_action_action
    num_next_state::Int
    num_state::Int
    num_action::Int
    evaluate_cache::Vector{T}
    jacobian_state_cache::Matrix{T}
    jacobian_action_cache::Matrix{T}
    hessian_prod_state_state_cache::Matrix{T}
    hessian_prod_action_state_cache::Matrix{T}
    hessian_prod_action_action_cache::Matrix{T}
end

Model{T} = Vector{Dynamics{T}} where T

function Dynamics(f::Function, num_state::Int, num_action::Int; quasi_newton::Bool=false)
    #TODO: option to load/save methods
    x = Symbolics.variables(:x, 1:num_state)
    u = Symbolics.variables(:u, 1:num_action)

    y = f(x, u)
    num_next_state = length(y)
    vx = Symbolics.variables(:vx, 1:num_next_state)  # vector variables for Hessian vector products
    
    jacobian_state = Symbolics.jacobian(y, x)
    jacobian_action = Symbolics.jacobian(y, u)
    evaluate_func = eval(Symbolics.build_function(y, x, u)[2])
    jacobian_state_func = eval(Symbolics.build_function(jacobian_state, x, u)[2])
    jacobian_action_func = eval(Symbolics.build_function(jacobian_action, x, u)[2])
    if !quasi_newton
        hessian_prod_state_state = sum([vx[t] .* Symbolics.hessian(y[t], x) for t in 1:num_next_state])
        hessian_prod_action_state = sum([vx[t] .* Symbolics.jacobian(Symbolics.gradient(y[t], u), x) for t in 1:num_next_state])
        hessian_prod_action_action = sum([vx[t] .* Symbolics.hessian(y[t], u) for t in 1:num_next_state])
        hessian_prod_state_state_func = eval(Symbolics.build_function(hessian_prod_state_state, x, u, vx)[2])
        hessian_prod_action_state_func = eval(Symbolics.build_function(hessian_prod_action_state, x, u, vx)[2])
        hessian_prod_action_action_func = eval(Symbolics.build_function(hessian_prod_action_action, x, u, vx)[2])
    else
        hessian_prod_state_state_func = nothing
        hessian_prod_action_state_func = nothing
        hessian_prod_action_action_func = nothing
    end

    return Dynamics(evaluate_func, jacobian_state_func, jacobian_action_func, hessian_prod_state_state_func,
                    hessian_prod_action_state_func, hessian_prod_action_action_func, num_next_state, num_state, num_action,
                    zeros(num_next_state), zeros(num_next_state, num_state), zeros(num_next_state, num_action),
                    zeros(num_state, num_state), zeros(num_action, num_state), zeros(num_action, num_action))
end

function dynamics!(d::Dynamics, cache, state, action)
    d.evaluate(cache, state, action)
    return nothing
end

function jacobian!(jacobian_states, jacobian_actions, dynamics::Vector{Dynamics{T}}, states, actions) where T
    for (t, d) in enumerate(dynamics)
        d.jacobian_state(d.jacobian_state_cache, states[t], actions[t])
        d.jacobian_action(d.jacobian_action_cache, states[t], actions[t])
        @views jacobian_states[t] .= d.jacobian_state_cache
        @views jacobian_actions[t] .= d.jacobian_action_cache
    end
end

function hessian_vector_prod!(hessian_prod_state_state, hessian_prod_action_state, hessian_prod_action_action,
        dynamics::Dynamics{T}, states, actions, parameters, lhs_vector) where T
    dynamics.hessian_prod_state_state(dynamics.hessian_prod_state_state_cache, states, actions, parameters, lhs_vector)
    dynamics.hessian_prod_action_state(dynamics.hessian_prod_action_state_cache, states, actions, parameters, lhs_vector)
    dynamics.hessian_prod_action_action(dynamics.hessian_prod_action_action_cache, states, actions, parameters, lhs_vector)
    @views hessian_prod_state_state .= dynamics.hessian_prod_state_state_cache
    @views hessian_prod_action_state .= dynamics.hessian_prod_action_state_cache
    @views hessian_prod_action_action .= dynamics.hessian_prod_action_action_cache
end

num_trajectory(dynamics::Vector{Dynamics{T}}) where T = sum([d.num_state + d.num_action for d in dynamics]) + dynamics[end].num_next_state

# user-provided dynamics and gradients
function Dynamics(f::Function, fx::Function, fu::Function, num_next_state::Int, num_state::Int, num_action::Int,
    ;fxx_prod::Function=nothing, fux_prod::Function=nothing, fuu_prod::Function=nothing)
    return Dynamics(f, fx, fu, fxx_prod, fux_prod, fuu_prod,
                    num_next_state, num_state, num_action,
                    zeros(num_next_state), zeros(num_next_state, num_state), zeros(num_next_state, num_action),
                    zeros(num_state, num_state), zeros(num_action, num_state), zeros(num_action, num_action))
end
