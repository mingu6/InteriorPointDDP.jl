"""
    Costs Data
"""

struct CostsData{C,X,U,XX,UU,UX}
    costs::C
    gradient_state::Vector{X}
    gradient_action::Vector{U}
    hessian_state_state::Vector{XX}
    hessian_action_action::Vector{UU}
    hessian_action_state::Vector{UX}
end

function costs_data(dynamics::Vector{Dynamics{T, J}}, costs) where {T, J}
	gradient_state = [[zeros(d.num_state) for d in dynamics]..., 
        zeros(dynamics[end].num_next_state)]
    gradient_action = [zeros(d.num_control) for d in dynamics]
    hessian_state_state = [[zeros(d.num_state, d.num_state) for d in dynamics]..., 
        zeros(dynamics[end].num_next_state, dynamics[end].num_next_state)]
    hessian_action_action = [zeros(d.num_control, d.num_control) for d in dynamics]
    hessian_action_state = [zeros(d.num_control, d.num_state) for d in dynamics]
    CostsData(costs, gradient_state, gradient_action, hessian_state_state, hessian_action_action, hessian_action_state)
end

function reset!(data::CostsData) 
    N = length(data.gradient_state) 
    for k = 1:N
        fill!(data.gradient_state[k], 0.0) 
        fill!(data.hessian_state_state[k], 0.0) 
        k == N && continue
        fill!(data.gradient_action[k], 0.0)
        fill!(data.hessian_action_action[k], 0.0)
        fill!(data.hessian_action_state[k], 0.0)
    end 
end