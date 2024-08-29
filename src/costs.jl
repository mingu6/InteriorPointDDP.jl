struct Cost
    evaluate
    gradient_state
    gradient_control
    hessian_state_state
    hessian_control_control
    hessian_control_state
end

function Cost(f::Function, num_state::Int, num_control::Int)
    x = Symbolics.variables(:x, 1:num_state)
    u = Symbolics.variables(:u, 1:num_control)

    evaluate = f(x, u)
    gradient_state = Symbolics.gradient(evaluate, x)
    gradient_control = Symbolics.gradient(evaluate, u)
    hessian_state_state = Symbolics.jacobian(gradient_state, x)
    hessian_control_control = Symbolics.jacobian(gradient_control, u)
    hessian_control_state = Symbolics.jacobian(gradient_control, x)

    gradient_state_func = eval(Symbolics.build_function(gradient_state, x, u)[2])
    gradient_control_func = eval(Symbolics.build_function(gradient_control, x, u)[2])
    hessian_state_state_func = eval(Symbolics.build_function(hessian_state_state, x, u)[2])
    hessian_control_control_func = eval(Symbolics.build_function(hessian_control_control, x, u)[2])
    hessian_control_state_func = eval(Symbolics.build_function(hessian_control_state, x, u)[2])

    return Cost(f, gradient_state_func, gradient_control_func,
        hessian_state_state_func, hessian_control_control_func, hessian_control_state_func)
end

Costs = Vector{Cost}

function cost(costs::Costs, states::Vector{Vector{T}}, controls::Vector{Vector{T}}) where T
    N = length(states)
    J = 0.0
    for t in 1:N-1
        J += costs[t].evaluate(states[t], controls[t])
    end
    J += costs[N].evaluate(states[N], T[])
    return J
end

function cost_gradient!(gradient_states::Vector{Vector{T}}, gradient_controls::Vector{Vector{T}}, costs::Costs,
            states::Vector{Vector{T}}, controls::Vector{Vector{T}}) where T
    N = length(costs)
    for t in 1:N-1
        costs[t].gradient_state(gradient_states[t], states[t], controls[t])
        costs[t].gradient_control(gradient_controls[t], states[t], controls[t])
    end
    costs[N].gradient_state(gradient_states[N], states[N], T[])
end

function cost_hessian!(hessian_state_state::Vector{Matrix{T}}, hessian_control_control::Vector{Matrix{T}},
            hessian_control_state::Vector{Matrix{T}}, costs::Costs, states::Vector{Vector{T}}, controls::Vector{Vector{T}}) where T
    N = length(costs)
    for t in 1:N-1
        costs[t].hessian_state_state(hessian_state_state[t], states[t], controls[t])
        costs[t].hessian_control_control(hessian_control_control[t], states[t], T[])
        costs[t].hessian_control_state(hessian_control_state[t], states[t], T[])
    end
    costs[N].hessian_state_state(hessian_state_state[N], states[N], T[])
end
