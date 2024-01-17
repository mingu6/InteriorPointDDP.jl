"""
    Constraints Data
"""
# TODO: consider splitting eq and inequality constraints apart rather than using indices_inequality to determine inequalities

struct ConstraintsData{T,C,CX,CU}
    constraints::Constraints{T}
    violations::Vector{C} # the current value of each constraint (includes equality and ineq.)
    jacobian_state::Vector{CX} 
    jacobian_action::Vector{CU}
    inequalities::Vector{Vector{T}} # inequality constraints only
    nominal_inequalities::Vector{Vector{T}}
    duals::Vector{Vector{T}} # duals (both eq and ineq) for each timestep # consider removing
    ineq_duals::Vector{Vector{T}} # only ineq duals for each timestep
    nominal_ineq_duals:: Vector{Vector{T}}
    slacks::Vector{Vector{T}}
    nominal_slacks::Vector{Vector{T}}
end

function constraint_data(model::Model, constraints::Constraints)
    H = length(constraints)
    c = [zeros(constraints[t].num_constraint) for t = 1:H]
    ineqs = [zeros(constraints[t].num_inequality) for t = 1:H]
    nominal_ineqs = [zeros(constraints[t].num_inequality) for t = 1:H]

    # take inequalities and package them together
    for t = 1:H
        @views c[t] .= constraints[t].evaluate_cache
        @views ineqs[t] .= constraints[t].evaluate_cache[constraints[t].indices_inequality] # cool indexing trick
    end
    
    cx = [zeros(constraints[t].num_constraint, t < H ? model[t].num_state : model[H-1].num_next_state) for t = 1:H]
    cu = [zeros(constraints[t].num_constraint, model[t].num_action) for t = 1:H-1]
    
    constraint_duals = [zeros(constraints[t].num_constraint) for t = 1:H]
    
    ineq_duals = [0.1 .* ones(constraints[t].num_inequality) for t = 1:H]
    nominal_ineq_duals = [0.1 .* ones(constraints[t].num_inequality) for t = 1:H]

    slacks = [0.01 .* ones(constraints[t].num_inequality) for t = 1:H]
    nominal_slacks = [0.01 .* ones(constraints[t].num_inequality) for t = 1:H]

    return ConstraintsData(constraints, c, cx, cu, ineqs, nominal_ineqs, constraint_duals, ineq_duals, nominal_ineq_duals, slacks, nominal_slacks)
end

function constraint!(constraint_data::ConstraintsData, x, u, w)
    constraint!(constraint_data.violations, constraint_data.inequalities, constraint_data.constraints, x, u, w)
end

function constraint_violation(constraint_data::ConstraintsData)  # TODO: needed????
    constraints = constraint_data.constraints
    H = length(constraints)
    max_violation = 0.0
    for t = 1:H
        num_constraint = constraints[t].num_constraint 
        ineq = constraints[t].indices_inequality
        for i = 1:num_constraint 
            c = constraint_data.violations[t][i]
            cti = (i in ineq) ? max(0.0, c) : abs(c)
            max_violation = max(max_violation, cti)
        end
    end
    return max_violation
end

function constraint_violation(constraint_data::ConstraintsData, x, u, w; 
    norm_type=Inf)
    constraint!(constraint_data, x, u, w)
    constraint_violation(constraint_data, norm_type=norm_type)
end

function constraint_violation_1norm(constr_data::ConstraintsData)
    y = constr_data.slacks
    c = constr_data.inequalities
    H = length(c)
    
    constr_violation = 0.
    for t = 1:H
        num_constraint = constr_data.constraints[t].num_inequality
        for i = 1:num_constraint
            constr_violation += abs(c[t][i] + y[t][i])
        end
    end
    return constr_violation
end
