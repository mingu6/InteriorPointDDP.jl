mutable struct InteriorPoint{T,C,CX,CU}
    objective::Objective{T}
    constraint_data::ConstraintsData{T,C,CX,CU}
    constraint_tmp::Vector{Vector{T}}
    constraint_jacobian_state_tmp::Vector{Matrix{T}} 
    constraint_jacobian_action_tmp::Vector{Matrix{T}}
end

function interior_point(model::Model{T}, objective::Objective{T}, constraints::Constraints{T}) where T
    # horizon
    H = length(model) + 1

    # pre-allocated memory

    # TODO: what does constraint_tmp do?
    constraint_tmp = [zeros(c.num_constraint) for c in constraints]
    constraint_jacobian_state_tmp = [zeros(c.num_constraint, t < H ? model[t].num_state : model[H-1].num_next_state) for (t, c) in enumerate(constraints)]
    constraint_jacobian_action_tmp = [zeros(c.num_constraint, t < H ? model[t].num_action : 0) for (t, c) in enumerate(constraints)]
    
    data = constraint_data(model, constraints)
    InteriorPoint(objective, 
        data, 
        constraint_tmp, 
        constraint_jacobian_state_tmp, 
        constraint_jacobian_action_tmp)
end

function cost(interior_point::InteriorPoint, states, actions, parameters)
    # objective
    J = cost(interior_point.objective, states, actions, parameters)

    # constraints
    constraint_data = interior_point.constraint_data
    c = constraint_data.violations

    # horizon
    H = length(c)

    constraint!(interior_point.constraint_data, states, actions, parameters)

    # s^T * c(x,u)
    for t = 1:H
        J += constraint_data.inequalities[t]' * constraint_data.ineq_duals[t]
    end

    return J
end

Base.length(objective::AugmentedLagrangianobjective) = length(objective.objective)