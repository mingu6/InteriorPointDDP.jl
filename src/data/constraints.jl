"""
    Constraints Data
"""

struct ConstraintsData{T,C,CX,CU,XX,UX,UU}
    constraints::Constraints{T}
    num_constraints::Vector{Int}
    num_ineq_lower::Vector{Int}
    num_ineq_upper::Vector{Int}
    residuals::Vector{C}
    ineq_lower::Vector{C}
    ineq_upper::Vector{C}
    jacobian_state::Vector{CX} 
    jacobian_action::Vector{CU}
    hessian_prod_state_state::Vector{XX}
    hessian_prod_action_state::Vector{UX}
    hessian_prod_action_action::Vector{UU}
end

function constraint_data(constraints::Constraints)
    num_constraints = [mapreduce(x -> x.num_constraint, +, constraints)]
    num_ineq_lower = [mapreduce(x -> x.num_ineq_lower, +, constraints)]
    num_ineq_upper = [mapreduce(x -> x.num_ineq_upper, +, constraints)]
    
    residuals = [zeros(h.num_constraint) for h in constraints]
    ineq_lower = [zeros(h.num_action) for h in constraints]
    ineq_upper = [zeros(h.num_action) for h in constraints]
    jac_x = [zeros(h.num_constraint, h.num_state) for h in constraints]
    jac_u = [zeros(h.num_constraint, h.num_action) for h in constraints]
    hessian_prod_state_state = [zeros(h.num_state, h.num_state) for h in constraints]
	hessian_prod_action_state = [zeros(h.num_action, h.num_state) for h in constraints]
	hessian_prod_action_action = [zeros(h.num_action, h.num_action) for h in constraints]
    
    return ConstraintsData(constraints, num_constraints, num_ineq_lower, num_ineq_upper,
            residuals, ineq_lower, ineq_upper, jac_x, jac_u,
            hessian_prod_state_state, hessian_prod_action_state, hessian_prod_action_action)
end

function reset!(data::ConstraintsData) 
    H = length(data.constraints) + 1
    data.num_constraints[1] = mapreduce(x -> x.num_constraint, +, data.constraints)
    data.num_ineq_lower[1] = mapreduce(x -> x.num_ineq_lower, +, data.constraints)
    data.num_ineq_upper[1] = mapreduce(x -> x.num_ineq_upper, +, data.constraints)
    for k = 1:H-1
        fill!(data.residuals[k], 0.0)
        fill!(data.ineq_lower[k], 0.0)
        fill!(data.ineq_upper[k], 0.0)
        fill!(data.jacobian_state[k], 0.0)
        fill!(data.jacobian_action[k], 0.0)
        fill!(data.hessian_prod_state_state[k], 0.0)
        fill!(data.hessian_prod_action_state[k], 0.0)
        fill!(data.hessian_prod_action_action[k], 0.0)
    end
end