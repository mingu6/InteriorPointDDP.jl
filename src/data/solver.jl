"""
    Solver Data
"""
mutable struct SolverData{T}
    costs::Vector{T}                    # cost value
    gradient::Vector{T}                 # Lagrangian gradient TODO: remove
    θ_max::T                            # filter initialization for maximum allowable constraint violation
    θ_min::T                            # minimum constraint violation for checking acceptable steps

    indices_state::Vector{Vector{Int}}  # indices for state trajectory  TODO: MAY NOT NEED THIS????
    indices_action::Vector{Vector{Int}} # indices for control trajectory

    step_size::Vector{T}                # step length
    status::Vector{Bool}                # solver status

    iterations::Vector{Int}

    cache::Dict{Symbol,Vector{T}}       # solver stats
    
    j::Int                              # outer iteration counter (i.e., j-th barrier subproblem)
    k::Int                              # overall iteration counter

    μ_j::Float64                        # perturbation value
    constr_viol_norm::Float64           # magnitude (1-norm) of constraint violation
    barrier_obj::Float64                # barrier objective function
    optimality_error::Float64           # optimality error for problem (not barrier)
    
    filter::Vector{Vector{T}}           # filter
end

function solver_data(dynamics::Vector{Dynamics{T}}; max_cache=1000) where T
    # indices x and u
    indices_state = Vector{Int}[]
    indices_action = Vector{Int}[] 
    n_sum = 0 
    m_sum = 0 
    n_total = sum([d.num_state for d in dynamics]) + dynamics[end].num_next_state
    for d in dynamics
        push!(indices_state, collect(n_sum .+ (1:d.num_state))) 
        push!(indices_action, collect(n_total + m_sum .+ (1:d.num_action)))
        n_sum += d.num_state 
        m_sum += d.num_action 
    end
    push!(indices_state, collect(n_sum .+ (1:dynamics[end].num_next_state)))

    costs = [Inf]
    θ_max = 0.0
    θ_min = 0.0
    step_size = [1.0]
    gradient = zeros(num_trajectory(dynamics))
    cache = Dict(:costs     => zeros(max_cache), 
                 :gradient      => zeros(max_cache), 
                 :θ_max => zeros(max_cache), 
                 :step_size     => zeros(max_cache))

    μ_j = 0.0
    constr_viol_norm = 0.0
    barrier_obj = 0.0
    optimality_error = 0.0
    filter = [[0.0 , 0.0]]

    SolverData(costs, gradient, θ_max, θ_min, indices_state, indices_action, step_size, [false], [0], cache, 1, 1, μ_j,
               constr_viol_norm, barrier_obj, optimality_error, filter)
end

function reset!(data::SolverData) 
    fill!(data.costs, 0.0) 
    fill!(data.gradient, 0.0)
    fill!(data.cache[:costs], 0.0) 
    fill!(data.cache[:gradient], 0.0) 
    fill!(data.cache[:step_size], 0.0) 
    data.θ_max = Inf
    data.θ_min = 0.0
    data.status[1] = false
    data.iterations[1] = 0
    data.j = 0
    data.k = 0
    data.μ_j = 0.0
    data.constr_viol_norm = 0.0
    data.barrier_obj = 0.0
    data.optimality_error = 0.0
    data.filter = [[zeros(2)]]
end

# TODO: fix iter
function cache!(data::SolverData)
    iter = 1 #data.cache[:iter] 
    # (iter > length(data[:costs])) && (@warn "solver data cache exceeded")
    data.cache[:costs][iter] = data.costs[1]
    data.cache[:gradient][iter] = data.gradient
    data.cache[:step_size][iter] = data.step_size
    data.cache[:j][iter] = data.j
    data.cache[:k][iter] = data.k
    data.cache[:μ_j][iter] = data.μ_j
    data.cache[:constr_viol_norm][iter] = data.constr_viol_norm
    data.cache[:barrier_obj][iter] = data.barrier_obj
    data.cache[:optimality_error][iter] = data.optimality_error
    data.cache[:filter][iter] = data.filter
    return nothing
end