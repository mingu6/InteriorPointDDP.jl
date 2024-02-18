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
		data.objective = cost(problem.costs.costs, problem.nominal_states, problem.nominal_actions, problem.parameters)
	elseif mode == :current
		data.objective = cost(problem.costs.costs, problem.states, problem.actions, problem.parameters)
	end
	return data.objective
end

function constraint!(problem::ProblemData; mode=:nominal)
    constr_data = problem.constraints
    states, actions, parameters = primal_trajectories(problem, mode=mode)
    inequalities, _, _ = dual_trajectories(constr_data, mode=mode)
    for (k, con) in enumerate(constr_data.constraints)
        con.num_constraint == 0 && continue
        con.evaluate(con.evaluate_cache, states[k], actions[k], parameters[k])
        @views constr_data.violations[k] .= con.evaluate_cache
        # take inequalities and package them together
        @views inequalities[k] .= con.evaluate_cache[con.indices_inequality]
        fill!(con.evaluate_cache, 0.0) # TODO: confirm this is necessary
    end
end

function barrier_objective!(problem::ProblemData, data::SolverData, feasible::Bool; mode=:nominal)
    N = problem.horizon
    constr_data = problem.constraints
    c, _, y = dual_trajectories(constr_data, mode=mode)
    
    barrier_obj = 0.
    for k = 1:N
        for i = constr_data.constraints[k].indices_inequality
            feasible ? barrier_obj -= log(-c[k][i]) : barrier_obj -= log(y[k][i])
        end
    end
    barrier_obj *= data.μ
    cost!(data, problem, mode=mode)
    barrier_obj += data.objective
    return barrier_obj
end

function constraint_violation_1norm(constr_data::ConstraintsData; mode=:nominal)
    # TODO: needs to include slack for IPDDP as well as equality
    c, _, y = dual_trajectories(constr_data, mode=mode)
    N = length(c)
    
    constr_violation = 0.
    for k = 1:N
        for i = constr_data.constraints[k].indices_inequality
            constr_violation += abs(c[k][i] + y[k][i])
        end
    end
    return constr_violation
end

function update_nominal_trajectory!(data::ProblemData, feasible::Bool) 
    N = data.horizon
    constraints = data.constraints
    for k = 1:N
        data.nominal_states[k] .= data.states[k]
        k == N && continue
        data.nominal_actions[k] .= data.actions[k]
        constraints.nominal_ineq_duals[k] .= constraints.ineq_duals[k]
        constraints.nominal_inequalities[k] .= constraints.inequalities[k]
        if !feasible
            constraints.nominal_slacks[k] .= constraints.slacks[k]
        end 
    end
end

function primal_trajectories(problem::ProblemData; mode=:nominal)
    x = mode == :nominal ? problem.nominal_states : problem.states
    u = mode == :nominal ? problem.nominal_actions : problem.actions
    w = problem.parameters
    return x, u, w 
end

function dual_trajectories(constr_data::ConstraintsData; mode=:nominal)
    c = mode == :nominal ? constr_data.nominal_inequalities : constr_data.inequalities
    s = mode == :nominal ? constr_data.nominal_ineq_duals : constr_data.ineq_duals
    y = mode == :nominal ? constr_data.nominal_slacks : constr_data.slacks
    return c, s, y
end
