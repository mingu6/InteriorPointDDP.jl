"""
    Problem Data
"""
mutable struct Solver{T}
    problem::ProblemData{T}
	policy::PolicyData{T}
	data::SolverData{T}
    options::Options{T}
end

function Solver(T, dynamics::Vector{Dynamics}, costs::Costs, constraints::Constraints,
            bounds::Union{Bounds, Nothing}=nothing; options::Union{Options, Nothing}=nothing)

    # allocate policy data  
    policy = policy_data(T, dynamics, constraints, bounds)

    # allocate model data
    problem = problem_data(T, dynamics, costs, constraints, bounds)

    # allocate solver data
    data = solver_data(T)

    options = isnothing(options) ? Options{T}() : options

	Solver(problem, policy, data, options)
end

function Solver(T, dynamics::Vector{Dynamics}, costs::Costs,
            bounds::Union{Bounds, Nothing}=nothing; options::Union{Options, Nothing}=nothing)
    N = length(costs)
    constraint = Constraint()
    constraints = [constraint for t = 1:N-1]
    
    # allocate policy data  
    policy = policy_data(T, dynamics, constraints)

    # allocate model data
    problem = problem_data(T, dynamics, costs, constraints, bounds)

    # allocate solver data
    data = solver_data(T)

    options = isnothing(options) ? Options{T}() : options

	Solver(T, problem, policy, data, options)
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

    for (t, uk) in enumerate(controls)
        # project primal variables within bounds

        bk = bounds[t]
        bkl = bounds[t].lower[bk.indices_lower]
        bku = bounds[t].upper[bk.indices_upper]

        ū[t] .= uk
        ū[t][bk.indices_lower] .= @views max.(uk[bk.indices_lower], bkl + options.κ_1 .* max.(1., bkl))
        ū[t][bk.indices_upper] .= @views min.(uk[bk.indices_upper], bku - options.κ_1 .* max.(1., bku))

        dynamics!(dynamics[t], x̄[t+1], x̄[t], ū[t])
    end
end

