function solve!(solver::Solver{T}, x1::Vector{T}, controls::Vector{Vector{T}}; kwargs...) where T
    initialize_trajectory!(solver, controls, x1)
    solve!(solver; kwargs...)
end

function solve!(solver::Solver{T}) where T
    (solver.options.verbose && solver.data.k==0) && solver_info()

	update_rule = solver.update_rule
    problem = solver.problem
    options = solver.options
	data = solver.data
    
    reset!(problem.model)
    reset!(problem.objective_data)
    reset!(data)
    reset_duals!(problem)
    
    time0 = time()
    
    # automatically select initial perturbation. loosely based on bound of CS condition (duality) for LPs
    fn_eval_time_ = time()
    objective!(data, problem, mode=:nominal)
    data.μ = options.μ_init

    constraint!(problem, data.μ; mode=:nominal)
    data.fn_eval_time += time() - fn_eval_time_
    
    # update performance measures for first iterate (req. for sufficient decrease conditions for step acceptance)
    data.primal_1_curr = constraint_violation_1norm(problem, mode=:nominal)
    data.barrier_obj_curr = barrier_objective!(problem, data, update_rule, mode=:nominal)
    
    # filter initialization for constraint violation and threshold for switching rule init. (step acceptance)
    data.max_primal_1 = 1e4 * max(1.0, data.primal_1_curr)
    data.min_primal_1 = 1e-4 * max(1.0, data.primal_1_curr)
    reset_filter!(data)
    
    num_bounds = sum(b.num_lower + b.num_upper for b in problem.bounds)

    while data.k < options.max_iterations
        fn_eval_time_ = time()
        evaluate_derivatives!(problem, mode=:nominal)
        data.fn_eval_time += time() - fn_eval_time_
        
        backward_pass!(update_rule, problem, data, options, mode=:nominal, verbose=options.verbose)
        data.status != 0 && break
        # check (outer) overall problem convergence

        data.dual_inf = dual_error(update_rule, problem, options)
        data.primal_inf = primal_error(problem)
        data.cs_inf = cs_error(update_rule, problem, options, T(0.0))
        
        # check (inner) barrier problem convergence and update barrier parameter if so

        cs_inf_μ = cs_error(update_rule, problem, options, data.μ)
        opt_err_μ = max(data.dual_inf, cs_inf_μ, data.primal_inf)
        opt_err_0 = max(data.dual_inf, data.cs_inf, data.primal_inf)

        opt_err_0 < options.optimality_tolerance && break

        if opt_err_μ <= options.κ_ϵ * data.μ && num_bounds > 0 && data.μ > options.optimality_tolerance / 10.0
            data.μ = max(options.optimality_tolerance / 10.0, min(options.κ_μ * data.μ, data.μ ^ options.θ_μ))
            reset_filter!(data)
            # performance of current iterate updated to account for barrier parameter change
            fn_eval_time_ = time()
            constraint!(problem, data.μ; mode=:nominal)
            data.fn_eval_time += time() - fn_eval_time_
            data.barrier_obj_curr = barrier_objective!(problem, data, update_rule, mode=:nominal)
            
            data.primal_1_curr = constraint_violation_1norm(problem, mode=:nominal)
            data.j += 1
            continue
        end
        
        options.verbose && iteration_status(data, options)
        
        forward_pass!(update_rule, problem, data, options, verbose=options.verbose)
        data.status != 0 && break
        
        update_nominal_trajectory!(problem)
        (!data.armijo_passed && !data.switching) && update_filter!(data, options)
        data.barrier_obj_curr = data.barrier_obj_next
        data.primal_1_curr = data.primal_1_next
        
        data.k += 1
        data.wall_time = time() - time0
        data.solver_time = data.wall_time - data.fn_eval_time
    end
    
    data.k == options.max_iterations && (data.status = 8)
    options.verbose && iteration_status(data, options)
    options.verbose && on_exit(data)
end

function update_filter!(data::SolverData{T}, options::Options{T}) where T
    new_filter_pt = [(1. - options.γ_θ) * data.primal_1_curr,
                        data.barrier_obj_curr - options.γ_φ * data.primal_1_curr]
    push!(data.filter, new_filter_pt)
end

function reset_filter!(data::SolverData{T}) where T
    empty!(data.filter)
    push!(data.filter, [data.max_primal_1, T(-Inf)])
    data.status = 0
end

function primal_error(problem::ProblemData{T}) where T
    N = problem.horizon

    primal_inf::T = 0   # constraint violation (primal infeasibility)
    for t = N:-1:1
        primal_inf = max(primal_inf, norm(problem.nominal_constraints[t], Inf))
    end
    return primal_inf
end

function dual_error(update_rule::UpdateRuleData{T}, problem::ProblemData{T},
                    options::Options{T}) where T
    N = problem.horizon
    bounds = problem.bounds
    ϕ, zl, zu, λ = dual_trajectories(problem, mode=:nominal)
    lu = problem.objective_data.gradient_control
    fu = problem.model.jacobian_control
    cu = problem.constraints_data.jacobian_control
    u_tmp1 = update_rule.u_tmp1

    num_ineq = 0
    z_norm::T = 0.0
    ϕ_norm::T = 0.0
    num_constr = problem.constraints_data.num_constraints[1]
    dual_inf::T = 0     # dual infeasibility (stationarity of Lagrangian of barrier subproblem)
    for t = N:-1:1
        u_tmp1[t] .= lu[t]
        mul!(u_tmp1[t], transpose(cu[t]), ϕ[t], 1.0, 1.0)
        u_tmp1[t] .-= zl[t]
        u_tmp1[t] .+= zu[t]
        t < N && mul!(u_tmp1[t], transpose(fu[t]), λ[t+1], 1.0, 1.0)
        dual_inf = max(dual_inf, norm(u_tmp1[t], Inf))

        z_norm += sum(zl[t])
        z_norm += sum(zu[t])
        ϕ_norm += norm(ϕ[t], 1)
        num_ineq += bounds[t].num_lower + bounds[t].num_upper
    end
    scaling = max(options.s_max, (ϕ_norm + z_norm) / max(num_ineq + num_constr, 1.0))  / options.s_max
    return dual_inf / scaling
end

function cs_error(update_rule::UpdateRuleData{T}, problem::ProblemData{T},
                    options::Options{T}, μ::T) where T
    u_tmp1 = update_rule.u_tmp1
    u_tmp2 = update_rule.u_tmp2
    N = problem.horizon
    bounds = problem.bounds
    _, _, _, il, iu = primal_trajectories(problem, mode=:nominal)
    _, zl, zu = dual_trajectories(problem, mode=:nominal)

    num_ineq::T = 0
    z_norm::T = 0
    cs_inf::T = 0     # dual infeasibility (stationarity of Lagrangian of barrier subproblem)
    for t = N:-1:1
        num_ineq += bounds[t].num_lower + bounds[t].num_upper
        (bounds[t].num_upper == 0 && bounds[t].num_lower == 0) && continue
        
        u_tmp1[t] .= il[t] 
        u_tmp1[t] .*= zl[t]
        u_tmp1[t] .-= μ
        replace!(u_tmp1[t], NaN=>0.0)
        cs_inf = max(cs_inf, norm(u_tmp1[t], Inf))
        u_tmp2[t] .= iu[t] 
        u_tmp2[t] .*= zu[t]
        u_tmp2[t] .-= μ
        replace!(u_tmp2[t], NaN=>0.0)
        cs_inf = max(cs_inf, norm(u_tmp2[t], Inf))
        z_norm += sum(zl[t])
        z_norm += sum(zu[t])
    end
    scaling = max(options.s_max, z_norm / max(num_ineq, 1.0))  / options.s_max
    return cs_inf / scaling
end

function reset_duals!(problem::ProblemData{T}) where T
    N = problem.horizon
    bounds = problem.bounds
    for t = 1:N
        fill!(problem.eq_duals[t], 0.0)
        fill!(problem.nominal_eq_duals[t], 0.0)
        fill!(problem.ineq_duals_lo[t], 0.0)
        fill!(problem.nominal_ineq_duals_lo[t], 0.0)
        fill!(problem.ineq_duals_up[t], 0.0)
        fill!(problem.nominal_ineq_duals_up[t], 0.0)
        fill!(problem.nominal_dyn_duals[t], 0.0)
        @views problem.ineq_duals_lo[t][bounds[t].indices_lower].= 1.0
        @views problem.ineq_duals_up[t][bounds[t].indices_upper] .= 1.0
        @views problem.nominal_ineq_duals_lo[t][bounds[t].indices_lower] .= 1.0
        @views problem.nominal_ineq_duals_up[t][bounds[t].indices_upper] .= 1.0
    end
end

