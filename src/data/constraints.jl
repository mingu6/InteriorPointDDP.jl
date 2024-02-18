"""
    Constraints Data
"""

struct ConstraintsData{T,C,CX,CU}
    constraints::Constraints{T}
    num_ineq::Vector{Int}
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

function constraint_data(model::Model, constraints::Constraints, ineq_dual_init::Float64, slack_init::Float64)
    N = length(constraints)
    c = [zeros(constraints[k].num_constraint) for k = 1:N]
    ineqs = [zeros(constraints[k].num_inequality) for k = 1:N]
    nominal_ineqs = [zeros(constraints[k].num_inequality) for k = 1:N]
    num_ineq = [mapreduce(x -> x.num_inequality, +, constraints)]

    # take inequalities and package them together
    for k = 1:N
        @views c[k] .= constraints[k].evaluate_cache
        @views ineqs[k] .= constraints[k].evaluate_cache[constraints[k].indices_inequality] # cool indexing trick
    end
    
    cx = [zeros(constraints[k].num_constraint, model[k].num_state) for k = 1:N-1]
    push!(cx, zeros(constraints[N].num_constraint, model[N-1].num_next_state))
    cu = [zeros(constraints[k].num_constraint, model[k].num_action) for k = 1:N-1]
    
    constraint_duals = [zeros(constraints[k].num_constraint) for k = 1:N]
    
    ineq_duals = [ineq_dual_init .* ones(constraints[k].num_inequality) for k = 1:N]
    nominal_ineq_duals = [ineq_dual_init .* ones(constraints[k].num_inequality) for k = 1:N]

    slacks = [slack_init .* ones(constraints[k].num_inequality) for k = 1:N]
    nominal_slacks = [slack_init .* ones(constraints[k].num_inequality) for k = 1:N]

    return ConstraintsData(constraints, num_ineq, c, cx, cu, ineqs, nominal_ineqs,
        constraint_duals, ineq_duals, nominal_ineq_duals, slacks, nominal_slacks)
end

function reset!(data::ConstraintsData, κ_1::Float64, κ_2::Float64) 
    N = length(data.constraints)
    data.num_ineq[1] = mapreduce(x -> x.num_inequality, +, data.constraints)
    for k = 1:N
        fill!(data.violations[k], 0.0)
        fill!(data.jacobian_state[k], 0.0)
        k < N && fill!(data.jacobian_action[k], 0.0)
        fill!(data.inequalities[k], 0.0)
        fill!(data.nominal_inequalities[k], 0.0)
        fill!(data.duals[k], 0.0)
        fill!(data.ineq_duals[k], κ_1) 
        fill!(data.nominal_ineq_duals[k], κ_1) 
        fill!(data.slacks[k], κ_2) 
        fill!(data.nominal_slacks[k], κ_2)
    end 
end