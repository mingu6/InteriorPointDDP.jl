"""
    Model Data
"""

struct ModelData{T,X,U,W,XX,XU,UU}
    dynamics::Vector{Dynamics{T}}
    jacobian_state::Vector{X}
    jacobian_action::Vector{U}
	jacobian_parameter::Vector{W} # TODO: Remove
    hessian_state_state::Vector{XX}
    hessian_state_action::Vector{XU}
    hessian_action_action::Vector{UU}
end

function model_data(dynamics::Vector{Dynamics{T}}) where T
	jacobian_state = [zeros(d.num_next_state, d.num_state) for d in dynamics]
    jacobian_action = [zeros(d.num_next_state, d.num_action) for d in dynamics]
	jacobian_parameter = [zeros(d.num_next_state, d.num_parameter) for d in dynamics]
    # TODO: Initialse hessians 
    ModelData(dynamics, jacobian_state, jacobian_action, jacobian_parameter)
end

function reset!(data::ModelData) 
    H = length(data.dynamics) + 1
    for t = 1:H-1 
        fill!(data.jacobian_state[t], 0.0) 
        fill!(data.jacobian_action[t], 0.0) 
        fill!(data.jacobian_parameter[t], 0.0) 
    end 
end 