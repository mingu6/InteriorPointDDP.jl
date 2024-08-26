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
    constr_data = problem.constr_data
    states, actions = primal_trajectories(problem, mode=mode)
    constr_traj = mode == :nominal ? problem.nominal_constraints : problem.constraints
    ineq_lo_traj = mode == :nominal ? problem.nominal_ineq_lower : problem.ineq_lower
    ineq_up_traj = mode == :nominal ? problem.nominal_ineq_upper : problem.ineq_upper
    for (k, con) in enumerate(constr_data.constraints)
        if con.num_constraint > 0
            con.evaluate(constr_traj[k], states[k], actions[k])
            for i in con.indices_compl
                constr_traj[k][i] -= μ
            end
        end
        evaluate_ineq_lower!(ineq_lo_traj[k], actions[k], con.bounds_lower)
        evaluate_ineq_upper!(ineq_up_traj[k], actions[k], con.bounds_upper)
        evaluate_cone_ineq!(ineq_up_traj[k], actions[k], con.cone_indices)
    end
end

function evaluate_ineq_lower!(res, actions, bound)
    m = length(actions)
    for i = 1:m
        res[i] = isinf(bound[i]) ? Inf : actions[i] - bound[i]
    end
end

function evaluate_ineq_upper!(res, actions, bound)
    m = length(actions)
    for i = 1:m
        res[i] = isinf(bound[i]) ? Inf : bound[i] - actions[i]
    end
end

function evaluate_cone_ineq!(res, actions, cone_inds)
    for inds in cone_inds
        if length(inds) == 0
            break
        end
        res[inds] .= actions[inds[1]] - norm(actions[inds[2:end]], 2)
    end
end

function barrier_objective!(problem::ProblemData, data::SolverData; mode=:nominal)
    N = problem.horizon
    constr_data = problem.constr_data
    _, _, h, il, iu = primal_trajectories(problem, mode=mode)
    ϕ = mode == :nominal ? problem.nominal_eq_duals : problem.eq_duals
    
    barrier_obj = 0.
    for k = 1:N-1
        # precompute constraint indices for speed
        constr = constr_data.constraints[k]
        for i = 1:constr.num_action
            if !isinf(il[k][i])
                barrier_obj -= log(il[k][i])
            end
            iscone = any(i in inds for inds in constr.cone_indices)
            if !isinf(iu[k][i]) && !iscone
                barrier_obj -= log(iu[k][i])
            end
        end
    end
    
    barrier_obj *= data.μ
    cost!(data, problem, mode=mode)
    barrier_obj += data.objective
    return barrier_obj
end

function constraint_violation_1norm(problem::ProblemData; mode=:nominal)
    _, _, h, _, _ = primal_trajectories(problem, mode=mode)
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
        data.nominal_ineq_lower[k] .= data.ineq_lower[k]
        data.nominal_ineq_upper[k] .= data.ineq_upper[k]
        data.nominal_eq_duals[k] .= data.eq_duals[k]
        data.nominal_ineq_duals_lo[k] .= data.ineq_duals_lo[k]
        data.nominal_ineq_duals_up[k] .= data.ineq_duals_up[k]
    end
end

function primal_trajectories(problem::ProblemData; mode=:nominal)
    x = mode == :nominal ? problem.nominal_states : problem.states
    u = mode == :nominal ? problem.nominal_actions : problem.actions
    h = mode == :nominal ? problem.nominal_constraints : problem.constraints
    il = mode == :nominal ? problem.nominal_ineq_lower : problem.ineq_lower
    iu = mode == :nominal ? problem.nominal_ineq_upper : problem.ineq_upper
    return x, u, h, il, iu 
end

function dual_trajectories(problem::ProblemData; mode=:nominal)
    ϕ = mode == :nominal ? problem.nominal_eq_duals : problem.eq_duals
    vl = mode == :nominal ? problem.nominal_ineq_duals_lo : problem.ineq_duals_lo
    vu = mode == :nominal ? problem.nominal_ineq_duals_up : problem.ineq_duals_up
    return ϕ, vl, vu
end
