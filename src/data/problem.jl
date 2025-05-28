mutable struct ProblemData{T}
    # current trajectory
    states::Vector{Vector{T}}
    controls::Vector{Vector{T}}
    constraints::Vector{Vector{T}}
    ineq_lo::Vector{Vector{T}}
    ineq_up::Vector{Vector{T}}
    eq_duals::Vector{Vector{T}}
    ineq_duals_lo::Vector{Vector{T}}
    ineq_duals_up::Vector{Vector{T}}
    dyn_duals::Vector{Vector{T}}

    # nominal trajectory
    nominal_states::Vector{Vector{T}}
    nominal_controls::Vector{Vector{T}}
    nominal_constraints::Vector{Vector{T}}
    nominal_ineq_lo::Vector{Vector{T}}
    nominal_ineq_up::Vector{Vector{T}}
    nominal_eq_duals::Vector{Vector{T}}
    nominal_ineq_duals_lo::Vector{Vector{T}}
    nominal_ineq_duals_up::Vector{Vector{T}}
    nominal_dyn_duals::Vector{Vector{T}}

    # model data
    model::ModelData{T}

    # objective/objective data
    objective_data::ObjectivesData{T}
    
    # constraints data
    constraints_data::ConstraintsData{T}

    # control bounds
    bounds::Bounds{T}

    horizon::Int
end

function problem_data(T, dynamics::Model, objectives::Objectives, constraints::Constraints, bounds::Union{Bounds, Nothing})

    @assert length(dynamics) + 1 == length(objectives) == length(constraints)
    @assert isnothing(bounds) ? length(bounds) == length(constraints) : true

	states = [zeros(T, c.num_state) for c in constraints]
    controls = [zeros(T, c.num_control) for c in constraints]
    constr = [zeros(T, c.num_constraint) for c in constraints]
    ineq_lo = deepcopy(controls)
    ineq_up = deepcopy(controls)
    eq_duals = deepcopy(constr)
    ineq_duals_lo = deepcopy(controls)
    ineq_duals_up = deepcopy(controls)
    dyn_duals = deepcopy(states)

    nominal_states = deepcopy(states)
    nominal_controls = deepcopy(controls)
    nominal_constr = deepcopy(constr)
    nominal_ineq_lo = deepcopy(controls)
    nominal_ineq_up = deepcopy(controls)
    nominal_eq_duals = deepcopy(constr)
    nominal_ineq_duals_lo = deepcopy(controls)
    nominal_ineq_duals_up = deepcopy(controls)
    nominal_dyn_duals = deepcopy(states)

    model = model_data(T, dynamics)
    objectives_dat = objectives_data(T, constraints, objectives)
    constr_data = constraint_data(T, constraints)
    bounds = isnothing(bounds) ? [Bound(T, c.num_control) for c in constraints] : bounds
    @assert all(length(b.lower) == length(b.upper) for b in bounds)
    @assert all(length(b.lower) == c.num_control for (b, c) in zip(bounds, constraints))
    
    horizon = length(objectives)
    
    ProblemData(states, controls, constr, ineq_lo, ineq_up,
        eq_duals, ineq_duals_lo, ineq_duals_up, dyn_duals,
        nominal_states, nominal_controls, nominal_constr, nominal_ineq_lo, nominal_ineq_up,
        nominal_eq_duals, nominal_ineq_duals_lo, nominal_ineq_duals_up, nominal_dyn_duals,
        model, objectives_dat, constr_data, bounds, horizon)
end
