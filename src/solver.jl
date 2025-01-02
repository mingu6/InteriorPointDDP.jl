"""
    Problem Data
"""
mutable struct Solver{T}
    problem::ProblemData{T}
    update_rule::UpdateRuleData{T}
	data::SolverData{T}
    options::Options{T}
end

function Solver(T, dynamics::Vector{Dynamics}, objectives::Objectives, constraints::Constraints,
            bounds::Union{Bounds, Nothing}=nothing; options::Union{Options, Nothing}=nothing)

    # allocate update rule data  
    update_rule = update_rule_data(T, dynamics, constraints, bounds)

    # allocate model data
    problem = problem_data(T, dynamics, objectives, constraints, bounds)

    # allocate solver data
    data = solver_data(T)

    options = isnothing(options) ? Options{T}() : options

	Solver(problem, update_rule, data, options)
end

function Solver(T, dynamics::Vector{Dynamics}, objectives::Objectives,
            bounds::Union{Bounds, Nothing}=nothing; options::Union{Options, Nothing}=nothing)
    N = length(objectives)
    constraint = Constraint()
    constraints = [constraint for t = 1:N-1]
    
    # allocate update rule data  
    update_rule = update_rule_data(T, dynamics, constraints)

    # allocate model data
    problem = problem_data(T, dynamics, objectives, constraints, bounds)

    # allocate solver data
    data = solver_data(T)

    options = isnothing(options) ? Options{T}() : options

	Solver(T, problem, update_rule, data, options)
end

function get_trajectory(solver::Solver{T}) where T
	return solver.problem.nominal_states, solver.problem.nominal_controls
end

function current_trajectory(solver::Solver{T}) where T
	return solver.problem.states, solver.problem.controls
end

function initialize_trajectory!(solver::Solver{T}, controls::Vector{Vector{T}}, x1::Vector{T}) where T
    bounds = solver.problem.bounds
    dynamics = solver.problem.model.dynamics
    options = solver.options

    solver.problem.nominal_states[1] .= x1
    ū = solver.problem.nominal_controls
    x̄ = solver.problem.nominal_states
    ūl = solver.problem.nominal_ineq_lo
    ūu = solver.problem.nominal_ineq_up

    u_tmp1 = solver.update_rule.u_tmp1
    u_tmp2 = solver.update_rule.u_tmp2

    for (t, ut) in enumerate(controls)
        # project primal variables within bounds

        # bt = bounds[t]
        # ūt = ū[t]
        # ūt .= ut

        # for i in bt.indices_lower
        #     ūt[i] = max(ut[i], bt.lower[i] + options.κ_1 * max(1.0, bt.lower[i]))
        # end

        # for i in bt.indices_upper
        #     ūt[i] = min(ut[i], bt.upper[i] - options.κ_1 * max(1.0, bt.upper[i]))
        # end

        u_tmp1[t] .= max.(bounds[t].lower, 1.0)
        # println("pppp t", t, " ", u_tmp1[t])
        u_tmp1[t] .*= options.κ_1
        u_tmp1[t] .+= bounds[t].lower
        ū[t] .= max.(controls[t], u_tmp1[t])

        u_tmp1[t] .= max.(bounds[t].upper, 1.0)
        u_tmp1[t] .*= -options.κ_1
        u_tmp1[t] .+= bounds[t].upper
        replace!(u_tmp1[t], NaN=>Inf)
        ū[t] .= min.(ū[t], u_tmp1[t])
        # initialise inequality constraints

        ūl[t] .= ū[t]
        ūl[t] .-= bounds[t].lower
        ūu[t] .= bounds[t].upper
        ūu[t] .-= ū[t]

        dynamics!(dynamics[t], x̄[t+1], x̄[t], ū[t])
    end
end

