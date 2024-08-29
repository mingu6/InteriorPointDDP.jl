"""
    Model Data
"""

struct ModelData{T,X,U,XX,UX,UU}
    dynamics::Vector{Dynamics{T}}
    jacobian_state::Vector{X}
    jacobian_action::Vector{U}
    hessian_prod_state_state::Vector{XX}
    hessian_prod_action_state::Vector{UX}
    hessian_prod_action_action::Vector{UU}
end

function model_data(dynamics::Vector{Dynamics{T}}) where T
	jacobian_state = [zeros(d.num_next_state, d.num_state) for d in dynamics]
    jacobian_action = [zeros(d.num_next_state, d.num_action) for d in dynamics]
	
	hessian_prod_state_state = [zeros(d.num_state, d.num_state) for d in dynamics]
	hessian_prod_action_state = [zeros(d.num_action, d.num_state) for d in dynamics]
	hessian_prod_action_action = [zeros(d.num_action, d.num_action) for d in dynamics]
    
    ModelData(dynamics, jacobian_state, jacobian_action,
            hessian_prod_state_state, hessian_prod_action_state, hessian_prod_action_action)
end

function reset!(data::ModelData) 
    N = length(data.dynamics) + 1
    for k = 1:N-1 
        fill!(data.jacobian_state[k], 0.0) 
        fill!(data.jacobian_action[k], 0.0) 
        fill!(data.hessian_prod_state_state[k], 0.0)
        fill!(data.hessian_prod_action_state[k], 0.0)
        fill!(data.hessian_prod_action_action[k], 0.0) 
    end 
end 