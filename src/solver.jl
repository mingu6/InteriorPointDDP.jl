"""
    Problem Data
"""
mutable struct Solver{T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O,FX,FU,FW,OX,OU,OXX,OUU,OUX,C,CX,CU}
    problem::ProblemData{T,X,U,D,O,FX,FU,FW,OX,OU,OXX,OUU,OUX,C,CX,CU}
	policy::PolicyData{N,M,NN,MM,MN,NNN,MNN}
	data::SolverData{T}
    options::Options{T}
end

function Solver(dynamics::Vector{Dynamics{T}}, costs::Costs{T}, constraints::Constraints{T};
    parameters=[[zeros(d.num_parameter) for d in dynamics]..., zeros(0)], options=Options{T}()) where T

    # allocate policy data  
    policy = policy_data(dynamics, constraints)

    # allocate model data
    problem = problem_data(dynamics, costs, constraints, options, parameters=parameters)

    # allocate solver data
    data = solver_data(dynamics)

	Solver(problem, policy, data, options)
end

function Solver(dynamics::Vector{Dynamics{T}}, costs::Costs{T};
    parameters=[[zeros(d.num_parameter) for d in dynamics]..., zeros(0)], options=Options{T}()) where T
    H = length(costs)
    constraint = Constraint()
    constraints = [constraint for t = 1:H]
    
    # allocate policy data  
    policy = policy_data(dynamics, constraints)

    # allocate model data
    problem = problem_data(dynamics, costs, constraints, options, parameters=parameters)

    # allocate solver data
    data = solver_data(dynamics)

	Solver(problem, policy, data, options)
end

function get_trajectory(solver::Solver)
	return solver.problem.nominal_states, solver.problem.nominal_actions[1:end-1]
end

function current_trajectory(solver::Solver)
	return solver.problem.states, solver.problem.actions[1:end-1]
end

function initialize_controls!(solver::Solver, actions) 
    for (t, ut) in enumerate(actions) 
        solver.problem.nominal_actions[t] .= ut
    end 
end

function initialize_states!(solver::Solver, states) 
    for (t, xt) in enumerate(states)
        solver.problem.nominal_states[t] .= xt
    end
end