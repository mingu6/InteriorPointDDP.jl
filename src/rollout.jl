function rollout!(policy::PolicyData, problem::ProblemData; feasible::Bool;
    step_size=1.0)

    if feasible
        return rollout_feasible!(policy, problem, step_size=1.0)
    else # infeasible
        return rollout_infeasible!(policy, problem, step_size=1.0)
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


function rollout_infeasible!(policy::PolicyData, problem::ProblemData;
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

    constraints = problem.objective.costs.constraint_data

    # policy
    K = policy.K
    k = policy.k

    Ks = policy.Ks
    ks = policy.ks

    tau = max(0.99, 1 - constraints.μ)

    ineq_duals = constraints.ineq_duals
    nominal_ineq_duals = constraints.nominal_ineq_duals

    slacks = constraints.slacks
    nominal_slacks = constraints.nominal_slacks # current slack variables
    
    Ky = policy.Ky
    ky = policy.ky

    for (t, d) in enumerate(dynamics)
        # s[t] .= s̄[t] + Ks[t] * (x[t] -x̄[t]) + step_size * ks[t]
        update!(ineq_duals[t], nominal_ineq_duals[t], Ks[t], ks[t], x[t], x̄[t], step_size)


        # y[t] .= ȳ[t] + Ky[t] * (x[t] -x̄[t]) + step_size * ky[t]
        update!(slacks[t], nominal_slacks[t], Ky[t], ky[t], x[t], x̄[t], step_size)

        # check slack and dual positivity
        num_ineq = constraints.constraints.num_inequality 
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

function rollout_feasible!(policy::PolicyData, problem::ProblemData; 
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

    constraints = problem.objective.costs.constraint_data

    # policy
    K = policy.K
    k = policy.k

    Ks = policy.Ks
    ks = policy.ks

    tau = max(0.99, 1 - constraints.μ)

    for (t, d) in enumerate(dynamics)
        # u[t] .= ū[t] + K[t] * (x[t] - x̄[t]) + step_size * k[t]
        update!(u[t], ū[t], K[t], k[t], x[t], x̄[t], step_size)

        # s[t] .= s̄[t] + Ks[t] * (x[t] -x̄[t]) + step_size * ks[t]
        update!(ineq_duals[t], nominal_ineq_duals[t], Ks[t], ks[t], x[t], x̄[t], step_size)
        
        num_ineq = constraints.constraints.num_inequality 
        # check dual variable positivity
        if check_positivity(ineq_duals[t], nominal_ineqs[t], num_ineq, tau) == false
            return false
        end

        # check strict constraint satisfaction
        if check_constr_sat(constraints.constraints[t], tau, states[t], actions[t], parameters[t]) == false
            return false
        end

        x[t+1] .= dynamics!(d, x[t], u[t], w[t])
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
    con_temp = constr
    if constr.num_constraint == 0
        return true
    end
    # otherwise, more than 1 constraint
    constr.evaluate(constr.evaluate_cache, state, action, parameters) # compute new constraint values
    for index in constr.indices_inequality
        if constr.evaluate_cache[index] > (1 - tau) * con_temp.violations[index] # calculations are stored in violations
            return false
        end
    end
    return true
end