function cost(problem::ProblemData; mode=:nominal)
    if mode == :nominal
        return cost(problem.costs.costs, problem.nominal_states, problem.nominal_actions, problem.parameters)
    elseif mode == :current
        return cost(problem.costs.costs, problem.states, problem.actions, problem.parameters)
    else 
        return 0.0 
    end
end

function cost!(data::SolverData, problem::ProblemData; mode=:nominal)
	if mode == :nominal
		data.costs[1] = cost(problem.costs.costs, problem.nominal_states, problem.nominal_actions, problem.parameters)
	elseif mode == :current
		data.costs[1] = cost(problem.costs.costs, problem.states, problem.actions, problem.parameters)
	end
	return data.costs
end

function barrier_objective!(problem::ProblemData, data::SolverData, feasible::Bool; mode=:nominal)
    H = problem.horizon
    constr_data = problem.constraints
    c = constr_data.inequalities
    y = constr_data.slacks

    barrier_obj = 0.
    if feasible
        for t = 1:H
            num_inequality = constr_data.constraints[t].num_inequality
            for i = 1:num_inequality
                barrier_obj -= log(-c[t][i])
            end
        end
    else
        for t = 1:H
            num_inequality = constr_data.constraints[t].num_inequality
            for i = 1:num_inequality
                barrier_obj -= log(y[t][i])
            end
        end
    end
    barrier_obj *= data.μ_j
    cost!(data, problem, mode=mode)
    barrier_obj += data.costs[1]
    return barrier_obj
end

function update_nominal_trajectory!(data::ProblemData, feasible::Bool) 
    H = data.horizon
    constraints = data.constraints
    
    for t = 1:H
        data.nominal_states[t] .= data.states[t]
        t == H && continue
        data.nominal_actions[t] .= data.actions[t]
        constraints.nominal_ineq_duals[t] .= constraints.ineq_duals[t]
        constraints.nominal_inequalities[t] .= constraints.inequalities[t]
        if !feasible
            constraints.nominal_slacks[t] .= constraints.slacks[t]
        end 
    end
end

function trajectories(problem::ProblemData; mode=:nominal) 
    x = mode == :nominal ? problem.nominal_states : problem.states
    u = mode == :nominal ? problem.nominal_actions : problem.actions
    w = problem.parameters
    return x, u, w 
end