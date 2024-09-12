function cost(problem::ProblemData{T}; mode=:nominal) where T
    if mode == :nominal
        return cost(problem.costs.costs, problem.nominal_states, problem.nominal_controls)
    elseif mode == :current
        return cost(problem.costs.costs, problem.states, problem.controls)
    else 
        return 0.0 
    end
end

function cost!(data::SolverData{T}, problem::ProblemData{T}; mode=:nominal) where T
	if mode == :nominal
		data.objective = cost(problem.cost_data.costs, problem.nominal_states, problem.nominal_controls)
	elseif mode == :current
		data.objective = cost(problem.cost_data.costs, problem.states, problem.controls)
	end
	return data.objective
end

function constraint!(problem::ProblemData{T}, μ::T; mode=:nominal) where T
    constraints = problem.constraints_data.constraints
    x, u, h = primal_trajectories(problem, mode=mode)
    for (t, con) in enumerate(constraints)
        if con.num_constraint > 0
            hk = h[t]
            con.evaluate(hk, x[t], u[t])
            for i in con.indices_compl
                hk[i] -= μ
            end
        end
    end
end

function barrier_objective!(problem::ProblemData{T}, data::SolverData{T}, policy::PolicyData{T}; mode=:nominal) where T
    N = problem.horizon
    bounds = problem.bounds
    u = mode == :nominal ? problem.nominal_controls : problem.controls

    bl1 = policy.bl_tmp1
    bu1 = policy.bu_tmp1
    
    barrier_obj = 0.
    for t = 1:N-1
        bt = bounds[t]

        bl1[t] .= @views u[t][bt.indices_lower]
        bl1[t] .-= @views bt.lower[bt.indices_lower]
        bl1[t] .= log.(bl1[t])

        bu1[t] .= @views bt.upper[bt.indices_upper]
        bu1[t] .-= @views u[t][bt.indices_upper]
        bu1[t] .= log.(bu1[t])

        barrier_obj -= sum(bl1[t])
        barrier_obj -= sum(bu1[t])
    end
    
    barrier_obj *= data.μ
    cost!(data, problem, mode=mode) # TODO: allocs
    barrier_obj += data.objective
    return barrier_obj
end

function constraint_violation_1norm(problem::ProblemData{T}; mode=:nominal) where T
    h = mode == :nominal ? problem.nominal_constraints : problem.constraints
    constr_violation = 0.
    for hk in h
        constr_violation += norm(hk, 1)
    end
    return constr_violation
end

function update_nominal_trajectory!(data::ProblemData) 
    N = data.horizon
    for t = 1:N
        data.nominal_states[t] .= data.states[t]
        t == N && continue
        data.nominal_controls[t] .= data.controls[t]
        data.nominal_constraints[t] .= data.constraints[t]
        data.nominal_eq_duals[t] .= data.eq_duals[t]
        data.nominal_ineq_duals_lo[t] .= data.ineq_duals_lo[t]
        data.nominal_ineq_duals_up[t] .= data.ineq_duals_up[t]
    end
end

function primal_trajectories(problem::ProblemData; mode=:nominal)
    x = mode == :nominal ? problem.nominal_states : problem.states
    u = mode == :nominal ? problem.nominal_controls : problem.controls
    h = mode == :nominal ? problem.nominal_constraints : problem.constraints
    return x, u, h
end

function dual_trajectories(problem::ProblemData; mode=:nominal)
    ϕ = mode == :nominal ? problem.nominal_eq_duals : problem.eq_duals
    vl = mode == :nominal ? problem.nominal_ineq_duals_lo : problem.ineq_duals_lo
    vu = mode == :nominal ? problem.nominal_ineq_duals_up : problem.ineq_duals_up
    return ϕ, vl, vu
end
