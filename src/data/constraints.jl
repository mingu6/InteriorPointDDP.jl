"""
    Constraints Data
"""

struct ConstraintsData{T}
    constraints::Constraints
    num_constraints::Vector{Int}
    residuals::Vector{Vector{T}}
    jacobian_state::Vector{Matrix{T}} 
    jacobian_control::Vector{Matrix{T}}
    vhux::Vector{Matrix{T}}  # DDP tensor contraction
    vhuu::Vector{Matrix{T}}  # DDP tensor contraction
end

function constraint_data(T, constraints::Constraints)
    num_constraints = [mapreduce(x -> x.num_constraint, +, constraints)]
    
    residuals = [zeros(T, h.num_constraint) for h in constraints]
    jac_x = [zeros(T, h.num_constraint, h.num_state) for h in constraints]
    jac_u = [zeros(T, h.num_constraint, h.num_control) for h in constraints]
	vhux = [zeros(T, h.num_control, h.num_state) for h in constraints]
	vhuu = [zeros(T, h.num_control, h.num_control) for h in constraints]
    
    return ConstraintsData(constraints, num_constraints, residuals,
            jac_x, jac_u, vhux, vhuu)
end

function reset!(data::ConstraintsData{T}) where T 
    N = length(data.constraints) + 1
    data.num_constraints[1] = mapreduce(x -> x.num_constraint, +, data.constraints)
    for t = 1:N-1
        fill!(data.residuals[t], 0.0)
        fill!(data.jacobian_state[t], 0.0)
        fill!(data.jacobian_control[t], 0.0)
        fill!(data.vhux[t], 0.0)
        fill!(data.vhuu[t], 0.0)
    end
end