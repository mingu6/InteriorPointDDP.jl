function rollout!(policy::PolicyData, problem::ProblemData, feasible::Bool, perturbation::Float64; step_size=1.0)
    if feasible
        return rollout_feasible!(policy, problem, perturbation; step_size)
    else # infeasible
        return rollout_infeasible!(policy, problem, perturbation; step_size)
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


function rollout_infeasible!(policy::PolicyData, problem::ProblemData, perturbation; step_size=1.0)
    # model 
    dynamics = problem.model.dynamics

    # trajectories
    states = problem.states
    actions = problem.actions
    params = problem.parameters

    constr_data = problem.constraints
    ineq_duals = constr_data.ineq_duals
    slacks = constr_data.slacks

    nominal_states = problem.nominal_states
    nominal_actions = problem.nominal_actions
    nominal_ineq_duals = constr_data.nominal_ineq_duals
    nominal_slacks = constr_data.nominal_slacks

    # initial state
    states[1] .= nominal_states[1]

    # policy
    Ku = policy.Ku
    ku = policy.ku

    Ks = policy.Ks
    ks = policy.ks

    Ky = policy.Ky
    ky = policy.ky

    tau = max(0.99, 1 - perturbation)

    for (t, d) in enumerate(dynamics)
        # s[t] .= s̄[t] + Ks[t] * (x[t] -x̄[t]) + step_size * ks[t]
        # update!(ineq_duals[t], nominal_ineq_duals[t], Ks[t], ks[t], states[t], nominal_states[t], step_size)
        ineq_duals[t] = nominal_ineq_duals[t] + step_size * ks[t] + Ks[t] * (states[t] - nominal_states[t])

        # y[t] .= ȳ[t] + Ky[t] * (x[t] -x̄[t]) + step_size * ky[t]
        # update!(slacks[t], nominal_slacks[t], Ky[t], ky[t], states[t], nominal_states[t], step_size)
        slacks[t] = nominal_slacks[t] + step_size * ky[t] + Ky[t] * (states[t] - nominal_states[t])

        # check slack and dual positivity  TODO: ensure indices inequality used so equality constraints can be handled
        num_ineq = constr_data.constraints[t].num_inequality 
        if check_positivity(ineq_duals[t], nominal_ineq_duals[t], num_ineq, tau) == false
            constr_data.ineq_duals .= copy(nominal_ineq_duals)
            constr_data.nominal_ineq_duals .= copy(nominal_ineq_duals)
            constr_data.slacks .= copy(nominal_slacks)
            constr_data.nominal_slacks .= copy(nominal_slacks)  # why reset to nominal slacks? these are just allocated memory for compute right?
            return false
        elseif check_positivity(slacks[t], nominal_slacks[t], num_ineq, tau) == false
            constr_data.ineq_duals .= copy(nominal_ineq_duals)
            constr_data.nominal_ineq_duals .= copy(nominal_ineq_duals)
            constr_data.slacks .= copy(nominal_slacks)
            constr_data.nominal_slacks .= copy(nominal_slacks)
            return false
        end
        
        # u[t] .= ū[t] + K[t] * (x[t] - x̄[t]) + step_size * k[t]
        # update!(actions[t], nominal_actions[t], Ku[t], ku[t], states[t], nominal_states[t], step_size)
        actions[t] = nominal_actions[t] + step_size * ku[t] + Ku[t] * (states[t] - nominal_states[t])
        # states[t+1] = dynamics!(d, states[t], actions[t], params[t])
        states[t+1] = copy(dynamics!(d, states[t], actions[t], params[t]))
    end        
    return true
end

function rollout_feasible!(policy::PolicyData, problem::ProblemData, perturbation; step_size=1.0)
    # model 
    dynamics = problem.model.dynamics

    # trajectories
    states = problem.states
    actions = problem.actions
    params = problem.parameters

    constr_data = problem.constraints
    ineq_duals = constr_data.ineq_duals
    
    nominal_states = problem.nominal_states # best states so far
    nominal_actions = problem.nominal_actions # best actions so far
    nominal_ineq_duals = constr_data.nominal_ineq_duals

    # initial state
    states[1] .= nominal_states[1]

    # policy
    Ku = policy.Ku
    ku = policy.ku

    Ks = policy.Ks
    ks = policy.ks

    tau = max(0.99, 1 - perturbation)

    constraints = constr_data.constraints
    violations = constr_data.violations

    for (t, d) in enumerate(dynamics)
        # u[t] .= ū[t] + K[t] * (x[t] - x̄[t]) + step_size * k[t]
        # update!(actions[t], nominal_actions[t], Ku[t], ku[t], states[t], nominal_states[t], step_size)
        actions[t] = nominal_actions[t] + step_size * ku[t] + Ku[t] * (states[t] - nominal_states[t])

        # s[t] .= s̄[t] + Ks[t] * (x[t] -x̄[t]) + step_size * ks[t]
        # update!(ineq_duals[t], nominal_ineq_duals[t], Ks[t], ks[t], states[t], nominal_states[t], step_size)
        ineq_duals[t] = nominal_ineq_duals[t] + step_size * ks[t] + Ks[t] * (states[t] - nominal_states[t])
        
        num_ineq = constr_data.constraints[t].num_inequality 
        # check dual variable positivity
        if check_positivity(ineq_duals[t], nominal_ineq_duals[t], num_ineq, tau) == false
            constr_data.ineq_duals .= copy(nominal_ineq_duals)
            constr_data.nominal_ineq_duals .= copy(nominal_ineq_duals)
            problem.actions .= copy(nominal_actions)
            problem.nominal_actions .= copy(nominal_actions)
            return false
        end

        # check strict constraint satisfaction
        if check_constr_sat!(constraints[t], violations[t], tau, states[t], actions[t], params[t]) == false
            constr_data.ineq_duals .= copy(nominal_ineq_duals)
            constr_data.nominal_ineq_duals .= copy(nominal_ineq_duals)
            problem.actions .= copy(nominal_actions)
            problem.nominal_actions .= copy(nominal_actions)
            return false
        end

        # states[t+1] .= dynamics!(d, states[t], actions[t], params[t])
        states[t+1] = copy(dynamics!(d, states[t], actions[t], params[t]))
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


function check_positivity(new, old, n, tau)
"""
    Assumes that the old vector already has positive values.
"""
    for i = 1:n
        if new[i] <  (1 - tau) * old[i]
            return false
        end
    end
    return true
end


function check_constr_sat!(constr::Constraint, violations, tau, state, action, parameters)
    if constr.num_constraint == 0
        return true
    end
    # otherwise, more than 1 constraint
    # old_vals = copy(violations) # TODO: check if this is needed
    constr.evaluate(constr.evaluate_cache, state, action, parameters) # compute new constraint values, not stored to violations yet
    indices_inequality = constr.indices_inequality
    if any(constr.evaluate_cache[indices_inequality] .> (1 - tau) .*  violations[indices_inequality])
        return false
    end
    return true
end