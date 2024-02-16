function solve!(solver::Solver{T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O}, args...; kwargs...) where {T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O<:Costs{T}}
    ipddp_solve!(solver, args...; kwargs...)
end

function ipddp_solve!(solver::Solver, states, actions; kwargs...)
    initialize_controls!(solver, actions)
    initialize_states!(solver, states)
    ipddp_solve!(solver; kwargs...)
end

function ipddp_solve!(solver::Solver; iteration=true)
    costs = []
    steps = []
    (solver.options.verbose && iteration==1) && solver_info()

	# data
	policy = solver.policy
    problem = solver.problem
    options = solver.options
    reset!(problem.model)
    reset!(problem.costs)
    reset!(problem.constraints, options.κ_1, options.κ_2)
	data = solver.data
    options.reset_cache && reset!(data)
    constr_data = solver.problem.constraints
    c = constr_data.inequalities
    s = constr_data.ineq_duals
    y = constr_data.slacks

    H = problem.horizon
    constraint!(constr_data, problem.nominal_states, problem.nominal_actions, problem.parameters)
    if options.feasible && any([any(ct .>= 0.0) for ct in c])
        options.feasible = false
        options.verbose && (@warn "Initialisation is infeasible, reverting to infeasible IPDDP.")
    end
    # update constraint evaluations for nominal trajectory
    if options.feasible
        for t = 1:H-1
            constr_data.nominal_inequalities[t] .= constr_data.inequalities[t]
        end
    end
    
    # initialize θ_max to initialise filter parameters for infeasible IPDDP
    θ_0 = options.feasible ? 0.0 : constraint_violation_1norm(constr_data)
    data.θ_max = !options.feasible ? 1e4 * max(1.0, θ_0) : 0.0
    data.θ_min = !options.feasible ? 1e-4 * max(1.0, θ_0) : 0.0
    data.constr_viol_norm = θ_0
    
    cost!(data, problem, mode=:nominal)[1]
    # automatically select initial perturbation. loosely based on bound of CS condition (duality) for LPs
    num_constraints = convert(Float64, sum([length(ct) for ct in c]))
    data.μ_j = (data.μ_j == 0.0) ? options.μ_0 * data.costs[1] / max(num_constraints, 1.0) : data.μ_j
    
    data.barrier_obj = barrier_objective!(problem, data, options.feasible, mode=:nominal)

    reset_filter!(data, options)

    time = 0.0
    while data.k <= options.max_iterations
        iter_time = @elapsed begin
            gradients!(problem, mode=:nominal)
            
            # check (outer) overall problem convergence
            opt_err = optimality_error(policy, problem, options, data.μ_j)
            data.optimality_error = opt_err
            
            # logging
            if options.verbose && data.k % 10 == 1
                println("")
                println(rpad("Iteration", 15), rpad("Time (s)", 15), rpad("Perturb. (μ)", 15), rpad("Cost", 15), rpad("Constr. Viol.", 15), 
                        rpad("Barrier Obj.", 15), rpad("Opt. Err.", 15), rpad("BP Reg.", 13), rpad("Step Size", 15))
            end
            if options.verbose
                println(
                    rpad(string(data.k), 15), 
                    rpad(@sprintf("%.5e", time), 15), 
                    rpad(@sprintf("%.5e", data.μ_j), 15), 
                    rpad(@sprintf("%.5e", data.costs[1]), 15), 
                    rpad(@sprintf("%.5e", data.constr_viol_norm), 15), 
                    rpad(@sprintf("%.5e", data.barrier_obj), 15), 
                    rpad(@sprintf("%.5e", data.optimality_error), 15), 
                    rpad(@sprintf("%.3e", data.ϕ_last), 13), 
                    rpad(@sprintf("%.5e", data.step_size[1]), 15)
                )            
            end 
            
            opt_err <= options.optimality_tolerance && break
            
            # check (inner) barrier problem convergence
            if opt_err <= options.κ_ϵ * data.μ_j
                data.μ_j = max(options.optimality_tolerance / 10.0, min(options.κ_μ * data.μ_j, data.μ_j ^ options.θ_μ))
                reset_filter!(data, options)
                data.barrier_obj = barrier_objective!(problem, data, options.feasible, mode=:nominal)
                data.j += 1
                data.k += 1
                continue
            end
            
            backward_pass!(policy, problem, data, options, verbose=options.verbose)
            !data.status[1] && break  # exit if backward pass fails
            
            constr_violation, barrier_obj, switching, armijo = forward_pass!(policy, problem, data, options, verbose=options.verbose)
            !data.status[1] && break  # exit if line search failed
            
            # accept trial step from forward pass! update nominal trajectory w/rollout
            options.feasible ? rescale_duals!(s, c, problem, data.μ_j, options::Options) : rescale_duals!(s, y, problem, data.μ_j, options::Options)
            update_nominal_trajectory!(problem, options.feasible)
            
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
            
            # update criteria for new iterate
            data.barrier_obj = barrier_obj
            data.constr_viol_norm = constr_violation
        end

        push!(costs, data.costs[1])
        push!(steps, data.step_size[1])
        data.k += 1
        time += iter_time
    end
    return nothing
end

function reset_filter!(data::SolverData, options::Options) 
    if !options.feasible
        data.filter = [[data.θ_max, -Inf]]
    else
        data.filter = [[0.0, Inf]]
    end
    data.status[1] = true
end

function optimality_error(policy::PolicyData, problem::ProblemData, options::Options, μ_j::Float64)
    stat_err::Float64 = 0   # stationarity of Lagrangian
    viol_err::Float64 = 0   # constraint violation (equality and slacks + ineq)
    cs_err::Float64 = 0     # complementary slackness
    s_norm::Float64 = 0     # optimality error rescaling term
    
    N = length(problem.states)
    constr_data = problem.constraints
    c = constr_data.nominal_inequalities
    s = constr_data.nominal_ineq_duals
    y = constr_data.nominal_slacks
    cy = options.feasible ? c : y
    !options.feasible && (μ_j *= -1)
    
    # Jacobians of system dynamics
    fx = problem.model.jacobian_state
    fu = problem.model.jacobian_action
    # Jacobian of inequality constraints 
    gx = constr_data.jacobian_state
    gu = constr_data.jacobian_action
    # Cost gradients
    lx = problem.costs.gradient_state
    lu = problem.costs.gradient_action
    
    policy.x_tmp[N] .= lx[N]
    
    for k = N-1:-1:1
        # TODO: document. Basically forget x and unroll derivative w.r.t. u recrusively
        policy.u_tmp[k] .= lu[k]
        mul!(policy.u_tmp[k], gu[k]', s[k], 1.0, 1.0)
        mul!(policy.u_tmp[k], fu[k]', policy.x_tmp[k+1], 1.0, 1.0)
        stat_err = max(stat_err, norm(policy.u_tmp[k], Inf))
        policy.x_tmp[k] .= lx[k]
        mul!(policy.x_tmp[k], gx[k]', s[k], 1.0, 1.0)
        mul!(policy.x_tmp[k], fx[k]', policy.x_tmp[k+1], 1.0, 1.0)
        
        num_inequality = constr_data.constraints[k].num_inequality
        for i = 1:num_inequality
            cs_err = max(cs_err, abs(s[k][i] * cy[k][i] + μ_j))
        end
        if !options.feasible
            for i = 1:num_inequality
                viol_err = max(viol_err, abs(c[k][i] + y[k][i]))
            end
        end
        s_norm += norm(s[k], 1)
    end
    
    num_constraints = convert(Float64, sum([length(ct) for ct in c]))
    s_d = max(options.s_max, s_norm / max(num_constraints, 1.0))  / options.s_max
    optimality_error = options.feasible ? max(stat_err / s_d, cs_err / s_d) : max(stat_err / s_d, viol_err, cs_err / s_d)
    return optimality_error
end
