function solve!(solver::Solver{T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O}, args...; kwargs...) where {T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O<:InteriorPoint{T}}
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
    
    # iteration counters
    j::Int = 1  # outer loop iteration
    k::Int = 1  # inner loop iteration (barrier sub-problem)
    l::Int = 1  # line search iteration

	# data
	policy = solver.policy
    problem = solver.problem
    reset!(problem.model)
    reset!(problem.objective)
	data = solver.data
    options = solver.options
    options.reset_cache && reset!(data)
    constraints = solver.problem.objective.costs.constraint_data

    # initial rollout is performed in caller file
	initial_cost = cost!(data, problem, mode=:nominal)
    constraint!(constraints, problem.nominal_states, problem.nominal_actions, problem.parameters)
	
	# initialize θ_max to initialise filter
	if !options.feasible
        slacks = problem.objective.costs.constraint_data.slacks
        data.θ_max = norm(constraints.violations + slacks, 1)
    else
        data.θ_max = Inf
    end
    
    if data.μ_j == 0
        initial_cost = initial_cost[1] # = data.objective[1] which is the obj func/cost for first iteration
        n_minus_1 = problem.horizon - 1
        num_inequals = constraints.constraints[1].num_inequality
        data.μ_j = initial_cost / n_minus_1 / num_inequals
    end

    reset_filter!(data, options)
    reset_regularisation!(data, options)

    time = 0
    while k <= options.max_iterations
        iter_time = @elapsed begin
            if k > 1
                constraint!(constraints, problem.nominal_states, problem.nominal_actions, problem.parameters)
            end
            gradients!(problem, mode=:nominal)
            backward_pass!(policy, problem, data, options)
            opt_err = optimality_error(policy, problem, options, 0.0)
            data.optimality_error = opt_err
            if opt_err <= options.optimality_tolerance  # converged!!! woohoo
                println("~~~~~~~~~~~~~~~~~~~")
                println("Optimality reached!")
                data.μ_j = 0.  # allows profiling, TODO: fix hack
                return nothing
            end
            opt_err_barrier = optimality_error(policy, problem, options, data.μ_j)
            if opt_err_barrier <= options.κ_ϵ * data.μ_j
                data.μ_j = max(options.optimality_tolerance / 10.0, min(options.κ_μ * data.μ_j, data.μ_j^options.θ_μ))
                reset_filter!(data, options)
                j += 1
                if k == 1
                    continue
                end
            end
            # when to update optimality error? when is filter updated?
            forward_pass!(policy, problem, data, options, min_step_size=options.min_step_size,
                    line_search=options.line_search, verbose=options.verbose)
            k += 1
        end
        # info
        data.iterations[1] += 1
        if k % 10 == 1
            println("")
            println(rpad("Iteration", 15), rpad("Elapsed time", 15), rpad("μ", 15), rpad("Cost", 15), rpad("Opt.error", 15), rpad("Reg.power", 13), rpad("Stepsize", 15))
        end
        if options.verbose
            println(
                rpad(string(k), 15), 
                rpad(@sprintf("%.5e", time+=iter_time), 15), 
                rpad(@sprintf("%.5e", data.μ_j), 15), 
                rpad(@sprintf("%.5e", data.objective[1]), 15), 
                rpad(@sprintf("%.5e", data.optimality_error), 15), 
                rpad(@sprintf("%.3e", options.reg), 13), 
                rpad(@sprintf("%.5e", data.step_size[1]), 15)
            )            
        end 

        push!(costs, data.objective[1])
        push!(steps, data.step_size[1])
    end
    
    data.μ_j = 0.  # allows profiling, TODO: fix hack
    return nothing
end

function reset_filter!(data::SolverData, options::Options) 
    if !options.feasible
        data.filter = [Inf, data.θ_max]
    else
        data.filter = [Inf, Inf]
    end
    data.status[1] = true
end

function reset_regularisation!(data::SolverData, options::Options)
    # TODO: Is this needed or not?
    options.start_reg = 0
    options.end_reg = 24 
    options.reg_step = 1
    data.status[1] = true
    options.recovery = 0.0
end

function optimality_error(policy::PolicyData, problem::ProblemData, options::Options, μ::Float64)
    stat_err::Float64 = 0   # stationarity of Lagrangian
    viol_err::Float64 = 0   # constraint violation (equality and slacks + ineq)
    cs_err::Float64 = 0     # complementary slackness
    s_norm::Float64 = 0     # optimality error rescaling term
    
    N = length(problem.states)
    Qu = policy.action_value.gradient_action
    constraints = problem.objective.costs.constraint_data
    s = constraints.ineq_duals
    y = constraints.slacks
    c = constraints.inequalities
    
    for t = 1:N-1
        stat_err = max(stat_err, norm(Qu[t], Inf))
        if options.feasible
            cs_err = max(cs_err, norm(s[t] .* c[t] .+ μ, Inf))
        else
            viol_err = max(viol_err, norm(c[t] + y[t], Inf))
            cs_err = max(cs_err, norm(s[t] .* y[t] .- μ, Inf))
        end
        s_norm += norm(s[t], 1)
    end
    
    s_max = options.s_max
    s_d = max(s_max, s_norm / (N * length(s[1])))  / s_max
    optimality_error = options.feasible ? max(stat_err / s_d, cs_err / s_d) : max(stat_err / s_d, viol_err, cs_err / s_d)
    return optimality_error
end
