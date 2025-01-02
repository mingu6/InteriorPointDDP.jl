"""
    Objectives Data
"""

struct ObjectivesData{T}
    objectives
    gradient_state::Vector{Vector{T}}
    gradient_control::Vector{Vector{T}}
    hessian_state_state::Vector{Matrix{T}}
    hessian_control_control::Vector{Matrix{T}}
    hessian_control_state::Vector{Matrix{T}}
end

function objectives_data(T, dynamics::Model, objectives::Objectives)
	gradient_state = [[zeros(T, d.num_state) for d in dynamics]..., 
        zeros(T, dynamics[end].num_next_state)]
    gradient_control = [zeros(T, d.num_control) for d in dynamics]
    hessian_state_state = [[zeros(T, d.num_state, d.num_state) for d in dynamics]..., 
        zeros(T, dynamics[end].num_next_state, dynamics[end].num_next_state)]
    hessian_control_control = [zeros(T, d.num_control, d.num_control) for d in dynamics]
    hessian_control_state = [zeros(T, d.num_control, d.num_state) for d in dynamics]
    ObjectivesData{T}(objectives, gradient_state, gradient_control, hessian_state_state, hessian_control_control, hessian_control_state)
end

function reset!(data::ObjectivesData{T}) where T 
    N = length(data.gradient_state) 
    for t = 1:N 
        fill!(data.gradient_state[t], 0.0) 
        fill!(data.hessian_state_state[t], 0.0) 
        t == N && continue
        fill!(data.gradient_control[t], 0.0)
        fill!(data.hessian_control_control[t], 0.0)
        fill!(data.hessian_control_state[t], 0.0)
    end 
end