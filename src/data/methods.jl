function objective(problem::ProblemData{T}; mode=:nominal) where T
    if mode == :nominal
        return objective(problem.objectives.objectives, problem.nominal_states, problem.nominal_controls)
    elseif mode == :current
        return objective(problem.objectives.objectives, problem.states, problem.controls)
    else 
        return 0.0 
    end
end

function objective!(data::SolverData{T}, problem::ProblemData{T}; mode=:nominal) where T
	if mode == :nominal
		data.objective = objective(problem.objective_data.objectives, problem.nominal_states, problem.nominal_controls)
	elseif mode == :current
		data.objective = objective(problem.objective_data.objectives, problem.states, problem.controls)
	end
	return data.objective
end

function constraint!(problem::ProblemData{T}, μ::T; mode=:nominal) where T
    constraints = problem.constraints_data.constraints
    x, u, c = primal_trajectories(problem, mode=mode)
    for (t, con) in enumerate(constraints)
        if con.num_constraint > 0
            ck = c[t]
            con.evaluate(ck, x[t], u[t])
            for i in con.indices_compl
                ck[i] -= μ
            end
        end
    end
end

function barrier_objective!(problem::ProblemData{T}, data::SolverData{T}, update_rule::UpdateRuleData{T}; mode=:nominal) where T
    N = problem.horizon
    bounds = problem.bounds
    _, _, _, il, iu = primal_trajectories(problem, mode=mode)
    u_tmp1 = update_rule.u_tmp1
    
    barrier_obj = 0.
    for t = 1:N

        u_tmp1[t] .= log.(il[t])
        for i in bounds[t].indices_lower
            barrier_obj -= u_tmp1[t][i]
        end

        u_tmp1[t] .= log.(iu[t])
        for i in bounds[t].indices_upper
            barrier_obj -= u_tmp1[t][i]
        end
    end
    
    barrier_obj *= data.μ
    fn_eval_time_ = time()
    objective!(data, problem, mode=mode)
    data.fn_eval_time += time() - fn_eval_time_
    barrier_obj += data.objective
    return barrier_obj
end

function constraint_violation_1norm(problem::ProblemData{T}; mode=:nominal) where T
    c = mode == :nominal ? problem.nominal_constraints : problem.constraints
    constr_violation = 0.
    for ck in c
        constr_violation += norm(ck, 1)
    end
    return constr_violation
end

function update_nominal_trajectory!(data::ProblemData) 
    N = data.horizon
    for t = 1:N
        data.nominal_states[t] .= data.states[t]
        data.nominal_controls[t] .= data.controls[t]
        data.nominal_constraints[t] .= data.constraints[t]
        data.nominal_ineq_lo[t] .= data.ineq_lo[t]
        data.nominal_ineq_up[t] .= data.ineq_up[t]
        data.nominal_eq_duals[t] .= data.eq_duals[t]
        data.nominal_ineq_duals_lo[t] .= data.ineq_duals_lo[t]
        data.nominal_ineq_duals_up[t] .= data.ineq_duals_up[t]
        data.nominal_dyn_duals[t] .= data.dyn_duals[t]
    end
end

function primal_trajectories(problem::ProblemData; mode=:nominal)
    x = mode == :nominal ? problem.nominal_states : problem.states
    u = mode == :nominal ? problem.nominal_controls : problem.controls
    c = mode == :nominal ? problem.nominal_constraints : problem.constraints
    il = mode == :nominal ? problem.nominal_ineq_lo : problem.ineq_lo
    iu = mode == :nominal ? problem.nominal_ineq_up : problem.ineq_up
    return x, u, c, il, iu
end

function dual_trajectories(problem::ProblemData; mode=:nominal)
    ϕ = mode == :nominal ? problem.nominal_eq_duals : problem.eq_duals
    zl = mode == :nominal ? problem.nominal_ineq_duals_lo : problem.ineq_duals_lo
    zu = mode == :nominal ? problem.nominal_ineq_duals_up : problem.ineq_duals_up
    λ = mode == :nominal ? problem.nominal_dyn_duals : problem.dyn_duals
    return ϕ, zl, zu, λ
end
