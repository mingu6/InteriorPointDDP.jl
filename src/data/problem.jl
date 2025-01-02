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

    # nominal trajectory
    nominal_states::Vector{Vector{T}}
    nominal_controls::Vector{Vector{T}}
    nominal_constraints::Vector{Vector{T}}
    nominal_ineq_lo::Vector{Vector{T}}
    nominal_ineq_up::Vector{Vector{T}}
    nominal_eq_duals::Vector{Vector{T}}
    nominal_ineq_duals_lo::Vector{Vector{T}}
    nominal_ineq_duals_up::Vector{Vector{T}}

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

    @assert length(dynamics) + 1 == length(objectives) == length(constraints) + 1
    @assert isnothing(bounds) ? length(bounds) == length(constraints) : true

	states = [[zeros(T, d.num_state) for d in dynamics]..., 
            zeros(T, dynamics[end].num_next_state)]
    controls = [zeros(T, d.num_control) for d in dynamics]
    constr = [zeros(T, h.num_constraint) for h in constraints]
    ineq_lo = [ones(T, d.num_control) for d in dynamics]
    ineq_up = [ones(T, d.num_control) for d in dynamics]
    eq_duals = [zeros(T, h.num_constraint) for h in constraints]
    ineq_duals_lo = [ones(T, d.num_control) for d in dynamics]
    ineq_duals_up = [ones(T, d.num_control) for d in dynamics]

    nominal_states = [[zeros(T, d.num_state) for d in dynamics]..., 
                zeros(T, dynamics[end].num_next_state)]
    nominal_controls = [zeros(T, d.num_control) for d in dynamics]
    nominal_constr = [zeros(T, h.num_constraint) for h in constraints]
    nominal_ineq_lo = [ones(T, d.num_control) for d in dynamics]
    nominal_ineq_up = [ones(T, d.num_control) for d in dynamics]
    nominal_eq_duals = [zeros(T, h.num_constraint) for h in constraints]
    nominal_ineq_duals_lo = [ones(T, d.num_control) for d in dynamics]
    nominal_ineq_duals_up = [ones(T, d.num_control) for d in dynamics]

    model = model_data(T, dynamics)
    objectives_dat = objectives_data(T, dynamics, objectives)
    constr_data = constraint_data(T, constraints)
    bounds = isnothing(bounds) ? [Bound(T, d.num_control) for d in dynamics] : bounds
    @assert all(length(b.lower) == length(b.upper) for b in bounds)
    @assert all(length(b.lower) == d.num_control for (b, d) in zip(bounds, dynamics))
    
    horizon = length(objectives)
    
    ProblemData(states, controls, constr, ineq_lo, ineq_up,
        eq_duals, ineq_duals_lo, ineq_duals_up,
        nominal_states, nominal_controls, nominal_constr, nominal_ineq_lo, nominal_ineq_up,
        nominal_eq_duals, nominal_ineq_duals_lo, nominal_ineq_duals_up,
        model, objectives_dat, constr_data, bounds, horizon)
end
