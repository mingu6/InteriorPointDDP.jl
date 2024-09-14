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

    for (t, ut) in enumerate(controls)
        # project primal variables within bounds
        bt = bounds[t]
        ūt = ū[t]
        ūt .= ut

        for i in bt.indices_lower
            ūt[i] = max(ut[i], bt.lower[i] + options.κ_1 * max(1.0, bt.lower[i]))
        end

        for i in bt.indices_upper
            ūt[i] = min(ut[i], bt.upper[i] - options.κ_1 * max(1.0, bt.upper[i]))
        end

        dynamics!(dynamics[t], x̄[t+1], x̄[t], ū[t])
    end
end

