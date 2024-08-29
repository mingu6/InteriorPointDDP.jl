function cost(problem::ProblemData; mode=:nominal)
    if mode == :nominal
        return cost(problem.costs.costs, problem.nominal_states, problem.nominal_actions)
    elseif mode == :current
        return cost(problem.costs.costs, problem.states, problem.actions)
    else 
        return 0.0 
    end
end

function cost!(data::SolverData, problem::ProblemData; mode=:nominal)
	if mode == :nominal
		data.objective = cost(problem.cost_data.costs, problem.nominal_states, problem.nominal_actions)
	elseif mode == :current
		data.objective = cost(problem.cost_data.costs, problem.states, problem.actions)
	end
	return data.objective
end

function constraint!(problem::ProblemData, μ::Float64; mode=:nominal)
    constraints = problem.constr_data.constraints
    bounds = problem.bounds
    x, u, h = primal_trajectories(problem, mode=mode)
    for (k, con) in enumerate(constraints)
        if con.num_constraint > 0
            hk = h[k]
            con.evaluate(hk, x[k], u[k])
            for i in con.indices_compl
                hk[i] -= μ
            end
        end
    end
end

function barrier_objective!(problem::ProblemData, data::SolverData; mode=:nominal)
    N = problem.horizon
    bounds = problem.bounds
    u = mode == :nominal ? problem.nominal_actions : problem.actions
    
    barrier_obj = 0.
    for k = 1:N-1
        bk = bounds[k]
        barrier_obj -= sum(log.(u[k][bk.indices_lower] - bk.lower[bk.indices_lower]))
        barrier_obj -= sum(log.(bk.upper[bk.indices_upper] - u[k][bk.indices_upper]))
    end
    
    barrier_obj *= data.μ
    cost!(data, problem, mode=mode)
    barrier_obj += data.objective
    return barrier_obj
end

function constraint_violation_1norm(problem::ProblemData; mode=:nominal)
    h = mode == :nominal ? problem.nominal_constraints : problem.constraints
    constr_violation = 0.
    for hk in h
        constr_violation += norm(hk, 1)
    end
    return constr_violation
end

function update_nominal_trajectory!(data::ProblemData) 
    N = data.horizon
    for k = 1:N
        data.nominal_states[k] .= data.states[k]
        k == N && continue
        data.nominal_actions[k] .= data.actions[k]
        data.nominal_constraints[k] .= data.constraints[k]
        data.nominal_eq_duals[k] .= data.eq_duals[k]
        data.nominal_ineq_duals_lo[k] .= data.ineq_duals_lo[k]
        data.nominal_ineq_duals_up[k] .= data.ineq_duals_up[k]
    end
end

function primal_trajectories(problem::ProblemData; mode=:nominal)
    x = mode == :nominal ? problem.nominal_states : problem.states
    u = mode == :nominal ? problem.nominal_actions : problem.actions
    h = mode == :nominal ? problem.nominal_constraints : problem.constraints
    return x, u, h
end

function dual_trajectories(problem::ProblemData; mode=:nominal)
    ϕ = mode == :nominal ? problem.nominal_eq_duals : problem.eq_duals
    vl = mode == :nominal ? problem.nominal_ineq_duals_lo : problem.ineq_duals_lo
    vu = mode == :nominal ? problem.nominal_ineq_duals_up : problem.ineq_duals_up
    return ϕ, vl, vu
end
