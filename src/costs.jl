struct Cost{T}
    evaluate
    gradient_state
    gradient_action
    hessian_state_state
    hessian_action_action
    hessian_action_state
    evaluate_cache::Vector{T}
    gradient_state_cache::Vector{T}
    gradient_action_cache::Vector{T}
    hessian_state_state_cache::Matrix{T}
    hessian_action_action_cache::Matrix{T}
    hessian_action_state_cache::Matrix{T}
end

function Cost(f::Function, num_state::Int, num_action::Int)
    x = Symbolics.variables(:x, 1:num_state)
    u = Symbolics.variables(:u, 1:num_action)

    evaluate = f(x, u)
    gradient_state = Symbolics.gradient(evaluate, x)
    gradient_action = Symbolics.gradient(evaluate, u)
    hessian_state_state = Symbolics.jacobian(gradient_state, x)
    hessian_action_action = Symbolics.jacobian(gradient_action, u)
    hessian_action_state = Symbolics.jacobian(gradient_action, x)

    evaluate_func = eval(Symbolics.build_function([evaluate], x, u)[2])
    gradient_state_func = eval(Symbolics.build_function(gradient_state, x, u)[2])
    gradient_action_func = eval(Symbolics.build_function(gradient_action, x, u)[2])
    hessian_state_state_func = eval(Symbolics.build_function(hessian_state_state, x, u)[2])
    hessian_action_action_func = eval(Symbolics.build_function(hessian_action_action, x, u)[2])
    hessian_action_state_func = eval(Symbolics.build_function(hessian_action_state, x, u)[2])

    return Cost(evaluate_func,
        gradient_state_func, gradient_action_func,
        hessian_state_state_func, hessian_action_action_func, hessian_action_state_func,
        zeros(1),
        zeros(num_state), zeros(num_action),
        zeros(num_state, num_state), zeros(num_action, num_action), zeros(num_action, num_state))
end

Costs{T} = Vector{Cost{T}} where T

function cost(costs::Vector{Cost{T}}, states, actions) where T
    J = 0.0
    for (t, cost) in enumerate(costs)
        cost.evaluate(cost.evaluate_cache, states[t], actions[t])
        J += cost.evaluate_cache[1]
    end
    return J
end

function cost_gradient!(gradient_states, gradient_actions, costs::Vector{Cost{T}}, states, actions) where T
    N = length(costs)
    for (k, cost) in enumerate(costs)
        cost.gradient_state(gradient_states[k], states[k], actions[k])
        k == N && continue
        cost.gradient_action(gradient_actions[k], states[k], actions[k])
    end
end

function cost_hessian!(hessian_state_state, hessian_action_action, hessian_action_state, costs::Vector{Cost{T}}, states, actions) where T
    N = length(costs)
    for (k, cost) in enumerate(costs)
        cost.hessian_state_state(hessian_state_state[k], states[k], actions[k])
        k == N && continue
        cost.hessian_action_action(hessian_action_action[k], states[k], actions[k])
        cost.hessian_action_state(hessian_action_state[k], states[k], actions[k])
    end
end
