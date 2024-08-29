"""
    Model Data
"""

struct ModelData{T}
    dynamics::Model
    jacobian_state::Vector{Matrix{T}}
    jacobian_control::Vector{Matrix{T}}
    vfxx::Vector{Matrix{T}}  # DDP tensor contraction
    vfux::Vector{Matrix{T}}  # DDP tensor contraction
    vfuu::Vector{Matrix{T}}  # DDP tensor contraction
end

function model_data(T, dynamics::Model)
	jacobian_state = [zeros(T, d.num_next_state, d.num_state) for d in dynamics]
    jacobian_control = [zeros(T, d.num_next_state, d.num_control) for d in dynamics]
	
	vfxx = [zeros(T, d.num_state, d.num_state) for d in dynamics]
	vfux = [zeros(T, d.num_control, d.num_state) for d in dynamics]
	vfuu = [zeros(T, d.num_control, d.num_control) for d in dynamics]
    
    ModelData{T}(dynamics, jacobian_state, jacobian_control, vfxx, vfux, vfuu)
end

function reset!(data::ModelData) 
    N = length(data.dynamics) + 1
    for t = 1:N-1 
        fill!(data.jacobian_state[t], 0.0) 
        fill!(data.jacobian_control[t], 0.0) 
        fill!(data.vfxx[t], 0.0)
        fill!(data.vfux[t], 0.0)
        fill!(data.vfuu[t], 0.0) 
    end 
end 