"""
    Constraints Data
"""

struct ConstraintsData{T,C,CX,CU,XX,UX,UU}
    constraints::Constraints{T}
    num_constraints::Vector{Int}
    residuals::Vector{C}
    jacobian_state::Vector{CX} 
    jacobian_action::Vector{CU}
    hessian_prod_state_state::Vector{XX}
    hessian_prod_action_state::Vector{UX}
    hessian_prod_action_action::Vector{UU}
end

function constraint_data(constraints::Constraints)
    num_constraints = [mapreduce(x -> x.num_constraint, +, constraints)]
    
    residuals = [zeros(h.num_constraint) for h in constraints]
    jac_x = [zeros(h.num_constraint, h.num_state) for h in constraints]
    jac_u = [zeros(h.num_constraint, h.num_control) for h in constraints]
    hessian_prod_state_state = [zeros(h.num_state, h.num_state) for h in constraints]
	hessian_prod_action_state = [zeros(h.num_control, h.num_state) for h in constraints]
	hessian_prod_action_action = [zeros(h.num_control, h.num_control) for h in constraints]
    
    return ConstraintsData(constraints, num_constraints,
            residuals, jac_x, jac_u,
            hessian_prod_state_state, hessian_prod_action_state, hessian_prod_action_action)
end

function reset!(data::ConstraintsData) 
    N = length(data.constraints) + 1
    data.num_constraints[1] = mapreduce(x -> x.num_constraint, +, data.constraints)
    for k = 1:N-1
        fill!(data.residuals[k], 0.0)
        fill!(data.jacobian_state[k], 0.0)
        fill!(data.jacobian_action[k], 0.0)
        fill!(data.hessian_prod_state_state[k], 0.0)
        fill!(data.hessian_prod_action_state[k], 0.0)
        fill!(data.hessian_prod_action_action[k], 0.0)
    end
end