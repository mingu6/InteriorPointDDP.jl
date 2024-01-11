function solve!(solver::Solver{T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O}, args...; kwargs...) where {T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O<:InteriorPoint{T}}
    ipddp_solve!(solver, args...; kwargs...)
end

function ipddp_solve!(solver::Solver, states, actions; kwargs...)
    initialize_controls!(solver, actions)
    initialize_states!(solver, states)
    ipddp_solve!(solver; kwargs...)
end

function ipddp_solve!(solver::Solver; iteration=true)

    # START TIMER
    costs = []
    steps = []
    (solver.options.verbose && iteration==1) && solver_info()

	# data
	policy = solver.policy
    problem = solver.problem
    reset!(problem.model)
    reset!(problem.objective)
	data = solver.data
    solver.options.reset_cache && reset!(data)
    options = solver.options
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
    
    if data.μⱼ == 0
        initial_cost = initial_cost[1] # = data.objective[1] which is the obj func/cost for first iteration
        n_minus_1 = problem.horizon - 1
        num_inequals = constraints.constraints[1].num_inequality
        data.μⱼ = initial_cost / n_minus_1 / num_inequals
    end

    reset_filter!(data, options)
    reset_regularisation!(data, options)

    time = 0
    for iter = 1:solver.options.max_iterations
        iter_time = @elapsed begin   
            gradients!(problem, mode=:nominal)
        backward_pass!(policy, problem, data, options)
        forward_pass!(policy, problem, data, options, min_step_size=solver.options.min_step_size,
            line_search=solver.options.line_search, verbose=solver.options.verbose)
        end 
    
        # info
        data.iterations[1] += 1
        if iter % 10 == 1
            println("")
            println(rpad("Iteration", 15), rpad("Elapsed time", 15), rpad("μ", 15), rpad("Cost", 15), rpad("Opt.error", 15), rpad("Reg.power", 13), rpad("Stepsize", 15))
        end
        if solver.options.verbose
            println(
                rpad(string(iter), 15), 
                rpad(@sprintf("%.5e", time+=iter_time), 15), 
                rpad(@sprintf("%.5e", data.μⱼ), 15), 
                rpad(@sprintf("%.5e", data.objective[1]), 15), 
                rpad(@sprintf("%.5e", options.opterr), 15), 
                rpad(@sprintf("%.3e", options.reg), 13), 
                rpad(@sprintf("%.5e", data.step_size[1]), 15)
            )            
        end 

        push!(costs, data.objective[1])
        push!(steps, data.step_size[1])

        # check convergence
        if max(options.opterr, data.μⱼ) <= options.objective_tolerance
            println("~~~~~~~~~~~~~~~~~~~")
            println("Optimality reached!")
            data.μⱼ = 0.  # allows profiling, TODO: fix hack
            return nothing
        end

        if options.opterr <= 0.2 * data.μⱼ
            data.μⱼ = max(options.objective_tolerance / 10.0, min(0.2 * data.μⱼ, data.μⱼ^1.2))
            reset_filter!(data, options)
        end
    end
    
    data.μⱼ = 0.  # allows profiling, TODO: fix hack
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