mutable struct ProblemData{T,X,U,D,O,FX,FU,FW,OX,OU,OXX,OUU,OUX,C,CX,CU}
    # current trajectory
    states::Vector{X}
    actions::Vector{U}

    # disturbance trajectory 
    # TODO: Ignore the disturbance trajectory
    parameters::Vector{D}

    # nominal trajectory
    nominal_states::Vector{X}
    nominal_actions::Vector{U}

    # model data
    model::ModelData{T,FX,FU,FW}

    # objective/cost data
    costs::CostsData{O,OX,OU,OXX,OUU,OUX}
    
    # constraints data
    constraints::ConstraintsData{T,C,CX,CU}

    # trajectory: z = (x1,..., xT, u1,..., uT-1) | Δz = (Δx1..., ΔxT, Δu1,..., ΔuT-1)
    trajectory::Vector{T}

    horizon::Int
end

function problem_data(dynamics, costs, constraints, options; parameters=[[zeros(d.num_parameter) for d in dynamics]..., zeros(0)])

    length(parameters) == length(dynamics) && (parameters = [parameters..., zeros(0)])
    @assert length(dynamics) + 1 == length(parameters)
    @assert length(dynamics) + 1 == length(costs)

	states = [[zeros(d.num_state) for d in dynamics]..., 
            zeros(dynamics[end].num_next_state)]
    actions = [[zeros(d.num_action) for d in dynamics]..., zeros(0)]

    nominal_states = [[zeros(d.num_state) for d in dynamics]..., 
            zeros(dynamics[end].num_next_state)]
    nominal_actions = [[zeros(d.num_action) for d in dynamics]..., zeros(0)]

    model = model_data(dynamics)
    costs_dat = costs_data(dynamics, costs)
    constr_data = constraint_data(dynamics, constraints, options.ineq_dual_init, options.slack_init)

    trajectory = zeros(num_trajectory(dynamics))
    
    horizon = length(costs)
    
    ProblemData(states, actions, parameters, nominal_states, nominal_actions, model, costs_dat, constr_data, trajectory, horizon)
end