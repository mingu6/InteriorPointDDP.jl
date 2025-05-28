"""
    Constraints Data
"""

struct ConstraintsData{T}
    constraints::Constraints
    num_constraints::Vector{Int}
    residuals::Vector{Vector{T}}
    jacobian_state::Vector{Matrix{T}} 
    jacobian_control::Vector{Matrix{T}}
    vcxx::Vector{Matrix{T}}  # DDP tensor contraction
    vcux::Vector{Matrix{T}}  # DDP tensor contraction
    vcuu::Vector{Matrix{T}}  # DDP tensor contraction
end

function constraint_data(T, constraints::Constraints)
    num_constraints = [mapreduce(x -> x.num_constraint, +, constraints)]
    
    residuals = [zeros(T, c.num_constraint) for c in constraints]
    jac_x = [zeros(T, c.num_constraint, c.num_state) for c in constraints]
    jac_u = [zeros(T, c.num_constraint, c.num_control) for c in constraints]
    vcxx = [zeros(T, c.num_state, c.num_state) for c in constraints]
	vcux = [zeros(T, c.num_control, c.num_state) for c in constraints]
	vcuu = [zeros(T, c.num_control, c.num_control) for c in constraints]
    
    return ConstraintsData(constraints, num_constraints, residuals,
            jac_x, jac_u, vcxx, vcux, vcuu)
end

function reset!(data::ConstraintsData{T}) where T 
    N = length(data.constraints)
    data.num_constraints[1] = mapreduce(x -> x.num_constraint, +, data.constraints)
    for t = 1:N
        fill!(data.residuals[t], 0.0)
        fill!(data.jacobian_state[t], 0.0)
        fill!(data.jacobian_control[t], 0.0)
        fill!(data.vcux[t], 0.0)
        fill!(data.vcuu[t], 0.0)
    end
end