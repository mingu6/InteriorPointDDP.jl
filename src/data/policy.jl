""" 
    Value function approximation 
"""
struct Value{T}
    gradient::Vector{Vector{T}}
    hessian::Vector{Matrix{T}}
end

""" 
    control-value function approximation 
"""
struct Hamiltonian{T}
    gradient_state::Vector{Vector{T}}
    gradient_control::Vector{Vector{T}}
    hessian_state_state::Vector{Matrix{T}}
    hessian_control_control::Vector{Matrix{T}}
    hessian_control_state::Vector{Matrix{T}}
end

"""
    Store all gains
"""
struct Gains{T}
    # data

    gains::Vector{Matrix{T}}
    gains_ineq::Vector{Matrix{T}}

    # views into feedforward/feedback

    # controls
    α::Vector{SubArray{T, 1, Matrix{T}, Tuple{UnitRange{Int64}, Int64}, true}}
    β::Vector{SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}}
    # dual eq.
    ψ::Vector{SubArray{T, 1, Matrix{T}, Tuple{UnitRange{Int64}, Int64}, true}}
    ω::Vector{SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}}
    # dual ineq.
    χl::Vector{SubArray{T, 1, Matrix{T}, Tuple{UnitRange{Int64}, Int64}, true}}
    ζl::Vector{SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}}
    χu::Vector{SubArray{T, 1, Matrix{T}, Tuple{UnitRange{Int64}, Int64}, true}}
    ζu::Vector{SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}}
end

"""
    Policy Data
"""
struct PolicyData{T}
    # policy u = ū + K * (x - x̄) + k
    gains_data::Gains{T}

    # value function approximation
    value::Value{T}

    # control-value (Q) function approximation
    hamiltonian::Hamiltonian{T}

    # pre-allocated memory
    x_tmp::Vector{Vector{T}}
    u_tmp::Vector{Vector{T}}
    h_tmp::Vector{Vector{T}}
	uu_tmp::Vector{Matrix{T}}
	ux_tmp::Vector{Matrix{T}}
    xx_tmp::Vector{Matrix{T}}
	hu_tmp::Vector{Matrix{T}}
	hx_tmp::Vector{Matrix{T}}

    bl_tmp1::Vector{Vector{T}}
    bl_tmp2::Vector{Vector{T}}
    bu_tmp1::Vector{Vector{T}}
    bu_tmp2::Vector{Vector{T}}

    lhs::Vector{Matrix{T}}
    lhs_tl::Vector{SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}}
    lhs_tr::Vector{SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}}
    lhs_bl::Vector{SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}}
    lhs_br::Vector{SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}}

    kkt_matrix_ws::Vector{BunchKaufmanWs{T}}
    D_cache::Vector{Pair{Vector{T}}}
end

function gains_data(T, constraints::Constraints)
    N = length(constraints) + 1

    gains = [zeros(T, c.num_control + c.num_constraint, c.num_state + 1) for c in constraints]
    gains_ineq = [zeros(T, 2 * c.num_control, c.num_state + 1) for c in constraints]

    α = [@views gains[t][1:constraints[t].num_control, 1] for t in 1:N-1]
    β = [@views gains[t][1:constraints[t].num_control, 2:end] for t in 1:N-1]

	ψ = [@views gains[t][constraints[t].num_control+1:end, 1] for t in 1:N-1]
    ω = [@views gains[t][constraints[t].num_control+1:end, 2:end] for t in 1:N-1]

    χl = [@views gains_ineq[t][1:constraints[t].num_control, 1] for t in 1:N-1]
    ζl = [@views gains_ineq[t][1:constraints[t].num_control, 2:end] for t in 1:N-1]
    
    χu = [@views gains_ineq[t][constraints[t].num_control+1:end, 1] for t in 1:N-1]
    ζu = [@views gains_ineq[t][constraints[t].num_control+1:end, 2:end] for t in 1:N-1]

    return Gains(gains, gains_ineq, α, β, ψ, ω, χl, ζl, χu, ζu)
end

function policy_data(T, dynamics::Vector{Dynamics}, constraints::Constraints, bounds::Bounds)
    gains = gains_data(T, constraints)

    # value function approximation
    P = [[zeros(T, d.num_state, d.num_state) for d in dynamics]..., 
            zeros(T, dynamics[end].num_next_state, dynamics[end].num_next_state)]
    p =  [[zeros(T, d.num_state) for d in dynamics]..., 
            zeros(T, dynamics[end].num_next_state)]
    value = Value(p, P)

    # control-value function approximation
    Qx = [zeros(T, d.num_state) for d in dynamics]
    Qu = [zeros(T, d.num_control) for d in dynamics]
    Qxx = [zeros(T, d.num_state, d.num_state) for d in dynamics]
    Quu = [zeros(T, d.num_control, d.num_control) for d in dynamics]
    Qux = [zeros(T, d.num_control, d.num_state) for d in dynamics]

    hamiltonian = Hamiltonian(Qx, Qu, Qxx, Quu, Qux)

    x_tmp = [[zeros(T, d.num_state) for d in dynamics]..., zeros(T, dynamics[end].num_next_state)]
    u_tmp = [zeros(T, d.num_control) for d in dynamics]
	h_tmp = [zeros(T, c.num_constraint) for c in constraints]
	uu_tmp = [zeros(T, d.num_control, d.num_control) for d in dynamics]
	ux_tmp = [zeros(T, d.num_control, d.num_state) for d in dynamics]
    xx_tmp = [zeros(T, d.num_state, d.num_state) for d in dynamics]

    hu_tmp = [zeros(T, c.num_constraint, c.num_control) for c in constraints]
    hx_tmp = [zeros(T, c.num_constraint, c.num_state) for c in constraints]

    lhs = [zeros(T, c.num_constraint + c.num_control, c.num_constraint + c.num_control) for c in constraints]
    lhs_tl = [@views lhs[t][1:c.num_control, 1:c.num_control] for (t, c) in enumerate(constraints)]
    lhs_tr = [@views lhs[t][1:c.num_control, c.num_control+1:end] for (t, c) in enumerate(constraints)]
    lhs_bl = [@views lhs[t][c.num_control+1:end, 1:c.num_control] for (t, c) in enumerate(constraints)]
    lhs_br = [@views lhs[t][c.num_control+1:end, c.num_control+1:end] for (t, c) in enumerate(constraints)]

    kkt_matrix_ws = [BunchKaufmanWs(L) for L in lhs]
    D_cache = [Pair(zeros(T, c.num_constraint + c.num_control), zeros(T, c.num_constraint + c.num_control)) for c in constraints]

    if !isnothing(bounds)
        bl_tmp1 = [zeros(T, length(b.indices_lower)) for b in bounds]
        bl_tmp2 = [zeros(T, length(b.indices_lower)) for b in bounds]
        bu_tmp1 = [zeros(T, length(b.indices_upper)) for b in bounds]
        bu_tmp2 = [zeros(T, length(b.indices_upper)) for b in bounds]
    else
        bl_tmp1 = [zeros(T, 0) for b in bounds]
        bl_tmp2 = [zeros(T, 0) for b in bounds]
        bu_tmp1 = [zeros(T, 0) for b in bounds]
        bu_tmp2 = [zeros(T, 0) for b in bounds]
    end

    PolicyData{T}(gains, value, hamiltonian,
        x_tmp, u_tmp, h_tmp, uu_tmp, ux_tmp, xx_tmp, hu_tmp, hx_tmp,
        bl_tmp1, bl_tmp2, bu_tmp1, bu_tmp2,
        lhs, lhs_tl, lhs_tr, lhs_bl, lhs_br, kkt_matrix_ws, D_cache)
end