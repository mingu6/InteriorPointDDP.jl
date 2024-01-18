function forward_pass!(policy::PolicyData, problem::ProblemData, data::SolverData, options::Options; #min_step_size=1.0e-5,
    verbose=false)
    data.status[1] = true
    data.step_size[1] = 1.0
    l = 1  # line search iteration
    min_step_size = -Inf

    μ_j = data.μ_j
    τ = max(options.τ_min, 1 - μ_j)  # fraction-to-boundary parameter
    constr_data = problem.constraints
    
    H = length(problem.states)
    c = constr_data.inequalities
    s = constr_data.ineq_duals
    y = constr_data.slacks
    c = constr_data.inequalities
    c̄ = constr_data.nominal_inequalities
    s̄ = constr_data.nominal_ineq_duals
    ȳ = constr_data.nominal_slacks

    while data.step_size[1] >= min_step_size # check whether we still want it to be this
        # generate proposed increment
        rollout!(policy, problem, options.feasible, step_size=data.step_size[1])
        # calibrate minimum step size
        if l == 1
            dir_deriv = trajectory_directional_derivative(problem, μ_j, options.feasible)
            if dir_deriv < 0.0 && data.constr_viol_norm <= data.θ_min
                min_step_size = min(options.γ_θ, -options.γ_φ * data.constr_viol_norm / dir_deriv,
                    options.δ * data.constr_viol_norm ^ options.s_θ / (-dir_deriv) ^ options.s_φ)
            elseif dir_deriv < 0.0 && data.constr_viol_norm > data.θ_min
                min_step_size = min(options.γ_θ, -options.γ_φ * data.constr_viol_norm / dir_deriv)
            else
                min_step_size = options.γ_θ
            end
            min_step_size *= options.γ_α
        end
        
        # check positivity using fraction-to-boundary condition on dual/slack variables (or constraints for feasible IPDDP)
        constraint!(constr_data, problem.states, problem.actions, problem.parameters)
        data.status[1] = check_positivity(s, s̄, problem, τ, false)
        data.status[1] = data.status[1] && (options.feasible ? check_positivity(c, c̄, problem, τ, true) : check_positivity(y, ȳ, problem, τ, false))
        !data.status[1] && (data.step_size[1] *= 0.5, l += 1, continue)  # failed, reduce step size
        
        # 1-norm of constraint violation of proposed step for filter (infeasible IPDDP only)
        constr_violation = options.feasible ? 0. : constraint_violation_1norm(constr_data)  
        
        # evaluate objective function of barrier problem to assess quality of iterate
        barrier_obj = barrier_objective!(problem, data, options.feasible, mode=:current)
        
        # check acceptability to filter A-5.4 IPOPT
        ind_replace_filter = 0
        for pt in data.filter
            if constr_violation >= pt[1] && barrier_obj >= pt[2]  # violation should stay 0. for fesible IPDDP
                data.status[1] = false
                break
            end
        end
        !data.status[1] && (data.step_size[1] *= 0.5, l += 1, continue)  # failed, reduce step size
        
        # additional checks for validity, e.g., switching condition, armijo or sufficient improvement w.r.t. filter
        dir_deriv = trajectory_directional_derivative(problem, μ_j, options.feasible)
        # if feasible, constraint violation not considered. just check armijo condition to accept trial point
        switching = ((dir_deriv < 0.0) && 
            ((- dir_deriv / data.step_size[1]) ^ options.s_φ * data.step_size[1] > options.δ * data.constr_viol_norm ^ options.s_θ))
        armijo = barrier_obj - data.barrier_obj <= options.η_φ * dir_deriv
        if (constr_violation <= data.θ_min) && switching
            data.status[1] = armijo
        else
            # sufficient progress conditions
            suff = !options.feasible ? (constr_violation <= (1. - options.γ_θ) * data.constr_viol_norm) : false
            suff = suff || (barrier_obj <= data.barrier_obj - options.γ_φ * data.constr_viol_norm)
            !suff && (data.status[1] = false)
        end
        !data.status[1] && (data.step_size[1] *= 0.5, l += 1, continue)  # failed, reduce step size
        
        # accept step!!! update nominal trajectory w/rollout TODO: move below out of forward pass to solve.jl
        update_nominal_trajectory!(problem, options.feasible)
        data.barrier_obj = barrier_obj
        data.constr_viol_norm = constr_violation
        # TODO: rescale dual variables if required (16)
        
        # check if filter should be augmented using accepted point
        if !armijo || !switching
            new_filter_pt = [(1. - options.γ_θ) * data.constr_viol_norm, data.barrier_obj - options.γ_φ * data.constr_viol_norm]
            # update filter by replacing existing point or adding new point
            filter_sz = length(data.filter)
            ind_replace_filter = 0
            for i in 1:filter_sz
                if new_filter_pt[1] <= data.filter[i][1] && new_filter_pt[2] <= data.filter[i][2]
                    ind_replace_filter = i
                    break
                end
            end
            ind_replace_filter == 0 ? push!(data.filter, new_filter_pt) : data.filter[ind_replace_filter] = new_filter_pt
        end
        break
    end
    !data.status[1] && (verbose && (@warn "Line search failed to find a suitable iterate"))
end

function check_positivity(s, s̄, problem::ProblemData, τ::Float64, flip::Bool)
    H = problem.horizon
    constr_data = problem.constraints
    if !flip
        for t = 1:H
            num_constraint = constr_data.constraints[t].num_inequality
            for i = 1:num_constraint
                s[t][i] < (1. - τ) *  s̄[t][i] && return false
            end
        end
    else
        for t = 1:H
            num_constraint = constr_data.constraints[t].num_inequality
            for i = 1:num_constraint
                s[t][i] > (1. - τ) *  s̄[t][i] && return false
            end
        end
    end
    return true
end

function trajectory_directional_derivative(problem::ProblemData, μ_j::Float64, feasible::Bool)
    H = problem.horizon
    # required to compute barrier objective gradient
    qx = problem.costs.gradient_state
    qu = problem.costs.gradient_action
    cx = problem.constraints.jacobian_state
    cu = problem.constraints.jacobian_action
    c = problem.constraints.inequalities
    y = problem.constraints.slacks
    
    # required to compute DDP increment
    x = problem.states
    u = problem.actions
    x̄ = problem.nominal_states
    ū = problem.nominal_actions
    ȳ = problem.constraints.nominal_slacks
    
    dir_deriv = 0.0  # grad. of barrier obj * DDP increment
    
    if feasible
        for t = 1:H-1
            num_inequalities = length(y[t])
            num_states = length(x[t])
            for i = 1:num_states
                dx = (x[t][i] - x̄[t][i])
                grad_obj_x_t = 0.0
                for j = 1:num_inequalities
                    grad_obj_x_t -= cx[t][j, i] / c[t][j]
                end
                grad_obj_x_t *= μ_j
                grad_obj_x_t += qx[t][i]
                dir_deriv += dx * grad_obj_x_t
            end
            num_actions = length(u[t])
            for i = 1:num_actions
                du = (u[t][i] - ū[t][i])
                grad_obj_u_t = 0.0
                for j = 1:num_inequalities
                    grad_obj_u_t += cu[t][j, i] / c[t][j]
                end
                grad_obj_u_t *= μ_j
                grad_obj_u_t += qx[t][i]
                dir_deriv += du * grad_obj_u_t
            end
        end
    else
        for t = 1:H
            num_states = length(x[t])
            for i = 1:num_states
                dir_deriv += (x[t][i] - x̄[t][i]) * qx[t][i]
            end
            num_actions = length(u[t])
            for i = 1:num_actions
                dir_deriv += (u[t][i] - ū[t][i]) * qu[t][i]
            end
            num_inequalities = length(y[t])
            for i = 1:num_inequalities
                dir_deriv -= (y[t][i] - ȳ[t][i]) * μ_j / y[t][i]
            end
        end
    end
    return dir_deriv
end
