function rollout!(policy::PolicyData, problem::ProblemData; feasible::Bool; perturbation::Float64;
    step_size=1.0)

    if feasible
        return rollout_feasible!(policy, problem, perturbation, step_size=1.0)
    else # infeasible
        return rollout_infeasible!(policy, problem, perturbation, step_size=1.0)
    end
end

    
function rollout(dynamics::Vector{Dynamics{T}}, initial_state, actions, 
    parameters=[zeros(d.num_parameter) for d in dynamics]) where T

    x_history = [initial_state]
    for (t, d) in enumerate(dynamics) 
        push!(x_history, copy(dynamics!(d, x_history[end], actions[t], parameters[t])))
    end
    
    return x_history
end


function rollout_infeasible!(policy::PolicyData, problem::ProblemData; perturbation;
    step_size=1.0)
    # model 
    dynamics = problem.model.dynamics

    # trajectories
    x = problem.states
    u = problem.actions
    w = problem.parameters
    x̄ = problem.nominal_states # old states
    ū = problem.nominal_actions # old actions

    # initial state
    x[1] .= x̄[1]

    constr_data = problem.objective.costs.constraint_data

    # policy
    K = policy.K
    k = policy.k

    Ks = policy.Ks
    ks = policy.ks

    tau = max(0.99, 1 - perturbation)

    ineq_duals = constr_data.ineq_duals
    nominal_ineq_duals = constr_data.nominal_ineq_duals

    slacks = constr_data.slacks
    nominal_slacks = constr_data.nominal_slacks # current slack variables
    
    Ky = policy.Ky
    ky = policy.ky

    for (t, d) in enumerate(dynamics)
        # s[t] .= s̄[t] + Ks[t] * (x[t] -x̄[t]) + step_size * ks[t]
        update!(ineq_duals[t], nominal_ineq_duals[t], Ks[t], ks[t], x[t], x̄[t], step_size)


        # y[t] .= ȳ[t] + Ky[t] * (x[t] -x̄[t]) + step_size * ky[t]
        update!(slacks[t], nominal_slacks[t], Ky[t], ky[t], x[t], x̄[t], step_size)

        # check slack and dual positivity
        num_ineq = constr_data.constraints.num_inequality 
        if check_positivity(ineq_duals[t], nominal_ineq_duals[t], num_ineq, tau) == false
            return false
        else if check_positivity(slacks[t], nominal_slacks[t], num_ineq, tau) == false
            return false
        end
        
        # u[t] .= ū[t] + K[t] * (x[t] - x̄[t]) + step_size * k[t]
        update!(u[t], ū[t], K[t], k[t], x[t], x̄[t], step_size)
        
        x[t+1] .= dynamics!(d, x[t], u[t], w[t])
    end        
    return true
end

function rollout_feasible!(policy::PolicyData, problem::ProblemData; perturbation;
    step_size=1.0)
    # model 
    dynamics = problem.model.dynamics

    # trajectories
    states = problem.states
    actions = problem.actions
    # params = problem.parameters
    nominal_states = problem.nominal_states # best states so far
    nominal_actions = problem.nominal_actions # best actions so far

    # initial state
    states[1] .= nominal_states[1]

    constr_data = problem.objective.costs.constraint_data

    # policy
    K = policy.K
    k = policy.k

    Ks = policy.Ks
    ks = policy.ks

    tau = max(0.99, 1 - perturbation)

    for (t, d) in enumerate(dynamics)
        # u[t] .= ū[t] + K[t] * (x[t] - x̄[t]) + step_size * k[t]
        update!(actions[t], nominal_actions[t], K[t], k[t], states[t], nominal_states[t], step_size)

        # s[t] .= s̄[t] + Ks[t] * (x[t] -x̄[t]) + step_size * ks[t]
        update!(ineq_duals[t], nominal_ineq_duals[t], Ks[t], ks[t], states[t], nominal_states[t], step_size)
        
        num_ineq = constr_data.constraints.num_inequality 
        # check dual variable positivity
        if check_positivity(ineq_duals[t], nominal_ineqs[t], num_ineq, tau) == false
            return false
        end

        constraints = constr_data.constraints
        # check strict constraint satisfaction
        if check_constr_sat(constraints[t], tau, states[t], actions[t], parameters[t]) == false
            return false
        end

        states[t+1] .= dynamics!(d, states[t], actions[t], w[t])
    end
    return true
end


function update!(vt, nominal_vt, Kt, kt, xt, nominal_xt, step_size)
    vt .= kt
    vt .*= step_size
    vt .+= nominal_vt
    mul!(vt, Kt, xt, 1.0, 1.0)
    mul!(vt, Kt, nominal_xt, -1.0 , 1.0)
end


function check_positivity(new; old, num_ineq, tau)
"""
    Assumes that the old vector already has positive values.
"""
    for i = 1:num_ineq
        if new[i] <  (1 - tau) * old[i]
            return false
        end
    end
    return true
end


function check_constr_sat(constr::Constraint, tau, state, action, parameters)
    if constr.num_constraint == 0
        return true
    end
    # otherwise, more than 1 constraint
    constr.evaluate(constr.evaluate_cache, state, action, parameters) # compute new constraint values, not stored to violations yet
    for index in constr.indices_inequality
        if constr.evaluate_cache[index] > (1 - tau) * constr.violations[index]
            return false
        end
    end
    return true
end