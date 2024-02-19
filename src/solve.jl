function solve!(solver::Solver{T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O}, args...; kwargs...) where {T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O<:Costs{T}}
    ipddp_solve!(solver, args...; kwargs...)
end

function ipddp_solve!(solver::Solver, states, actions; kwargs...)
    initialize_controls!(solver, actions)
    initialize_states!(solver, states)
    ipddp_solve!(solver; kwargs...)
end

function ipddp_solve!(solver::Solver)
    (solver.options.verbose && solver.data.k==0) && solver_info()
	
	policy = solver.policy
    problem = solver.problem
    options = solver.options
	data = solver.data
    constr_data = problem.constraints
    
    reset!(problem.model)
    reset!(problem.costs)
    reset!(problem.constraints, options.ineq_dual_init, options.slack_init)
    options.reset_cache && reset!(data)

    constraint!(problem, mode=:nominal)
    if options.feasible && any(map(x -> any(x .>= 0.0), constr_data.nominal_inequalities))
        options.feasible = false
        options.verbose && (@warn "Initialisation is infeasible, reverting to infeasible IPDDP.")
    end
    
    # automatically select initial perturbation. loosely based on bound of CS condition (duality) for LPs
    cost!(data, problem, mode=:nominal)
    data.μ = (data.μ == 0.0) ? options.μ_init * data.objective / max(constr_data.num_ineq[1], 1.0) : data.μ
    
    # update performance measures for first iterate (req. for sufficient decrease conditions for step acceptance)
    data.primal_1_curr = options.feasible ? 0.0 : constraint_violation_1norm(constr_data, mode=:nominal)
    data.barrier_obj_curr = barrier_objective!(problem, data, options.feasible, mode=:nominal)
    
    # filter initialization for constraint violation and threshold for switching rule init. (step acceptance)
    data.max_primal_1 = !options.feasible ? 1e4 * max(1.0, data.primal_1_curr) : 0.0
    data.min_primal_1 = !options.feasible ? 1e-4 * max(1.0, data.primal_1_curr) : 0.0
    reset_filter!(data, options)

    while data.k < options.max_iterations
        iter_time = @elapsed begin
            gradients!(problem, mode=:nominal)
            
            # check (outer) overall problem convergence
            data.dual_inf, data.primal_inf, data.cs_inf = optimality_error(policy, problem, options, 0.0, mode=:nominal)
            opt_err_0 = max(data.dual_inf, data.cs_inf)
            !options.feasible && (opt_err_0 = max(opt_err_0, data.primal_inf))
            
            opt_err_0 <= options.optimality_tolerance && break
            
            # check (inner) barrier problem convergence and update barrier parameter if so
            dual_inf_μ, primal_inf_μ, cs_inf_μ = optimality_error(policy, problem, options, data.μ, mode=:nominal)
            opt_err_μ = max(dual_inf_μ, cs_inf_μ)
            !options.feasible && (opt_err_μ = max(opt_err_μ, primal_inf_μ))            
            if opt_err_μ <= options.κ_ϵ * data.μ
                data.μ = max(options.optimality_tolerance / 10.0, min(options.κ_μ * data.μ, data.μ ^ options.θ_μ))
                reset_filter!(data, options)
                # performance of current iterate updated to account for barrier parameter change
                data.barrier_obj_curr = barrier_objective!(problem, data, options.feasible, mode=:nominal)
                data.j += 1
                continue
            end
            
            options.verbose && iteration_status(data, options)
            
            backward_pass!(policy, problem, data, options, mode=:nominal, verbose=options.verbose)
            !data.status && break
            
            forward_pass!(policy, problem, data, options, verbose=options.verbose)
            !data.status && break
            
            rescale_duals!(constr_data, data.μ, options)
            update_nominal_trajectory!(problem, options.feasible)
            update_filter!(data, options)
            data.barrier_obj_curr = data.barrier_obj_next
            data.primal_1_curr = data.primal_1_next
        end
        
        data.k += 1
        data.wall_time += iter_time
    end
    
    options.verbose && iteration_status(data, options)
    if data.k == options.max_iterations 
        data.status = false
        options.verbose && @warn "Maximum solver iterations reached."
    end
    
    return nothing
end

function update_filter!(data::SolverData, options::Options)
    if !data.armijo_passed || !data.switching
        new_filter_pt = [(1. - options.γ_θ) * data.primal_1_curr,
                         data.barrier_obj_curr - options.γ_φ * data.primal_1_curr]
        push!(data.filter, new_filter_pt)
    end
end

function reset_filter!(data::SolverData, options::Options)
    empty!(data.filter)
    options.feasible ? push!(data.filter, [0.0, Inf]) : push!(data.filter, [data.max_primal_1, -Inf])
    data.status = true
end

function optimality_error(policy::PolicyData, problem::ProblemData, options::Options, μ::Float64; mode=:nominal)
    dual_inf::Float64 = 0     # dual infeasibility (stationarity of Lagrangian)
    primal_inf::Float64 = 0   # constraint violation (primal infeasibility)
    cs_inf::Float64 = 0       # complementary slackness violation
    s_norm::Float64 = 0       # optimality error rescaling term
    
    N = problem.horizon
    constr_data = problem.constraints
    c, s, y = dual_trajectories(constr_data, mode=mode)
    
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
        dual_inf = max(dual_inf, norm(policy.u_tmp[k], Inf))
        policy.x_tmp[k] .= lx[k]
        mul!(policy.x_tmp[k], gx[k]', s[k], 1.0, 1.0)
        mul!(policy.x_tmp[k], fx[k]', policy.x_tmp[k+1], 1.0, 1.0)
        
        for i = constr_data.constraints[k].indices_inequality
            if options.feasible
                cs_inf = max(cs_inf, abs(s[k][i] * c[k][i] + μ))
            else
                cs_inf = max(cs_inf, abs(s[k][i] * y[k][i] - μ))
                primal_inf = max(primal_inf, abs(c[k][i] + y[k][i]))
            end
        end
        s_norm += norm(s[k], 1)
    end
    
    scaling = max(options.s_max, s_norm / max(constr_data.num_ineq[1], 1.0))  / options.s_max
    return dual_inf / scaling, primal_inf, cs_inf / scaling
end

function rescale_duals!(constr_data::ConstraintsData, μ::Float64, options::Options)
    N = length(constr_data.constraints)
    κ_Σ = options.κ_Σ
    options.feasible && (μ *= -1.0)
    c = constr_data.inequalities
    s = constr_data.ineq_duals
    y = constr_data.slacks
    cy = options.feasible ? c : y
    for k = 1:N
        for i = constr_data.constraints[k].indices_inequality
            s[k][i] = max(min(s[k][i], κ_Σ * μ / cy[k][i]), μ / (κ_Σ *  cy[k][i]))
        end
    end
end
