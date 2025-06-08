""" 
    Value function approximation 
"""
struct Value{T}
    gradient::Vector{Vector{T}}
    hessian::Vector{Matrix{T}}
end

"""
    Store all update rule parameters
"""
struct UpdateRuleParameters{T}
    # data
    eq::Vector{Matrix{T}}
    ineq::Vector{Matrix{T}}

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
    Update Rule Data
"""
struct UpdateRuleData{T}
    parameters::UpdateRuleParameters{T}

    # value function approximation
    value::Value{T}

    # intermediate variables for value function
    Qû::Vector{Vector{T}}
    C::Vector{Matrix{T}}
    Ĥ::Vector{Matrix{T}}
    B::Vector{Matrix{T}}

    # pre-allocated memory
    x_tmp::Vector{Vector{T}}
    u_tmp1::Vector{Vector{T}}
    u_tmp2::Vector{Vector{T}}
    c_tmp::Vector{Vector{T}}
	uu_tmp::Vector{Matrix{T}}
	ux_tmp::Vector{Matrix{T}}
    xx_tmp::Vector{Matrix{T}}
	cu_tmp::Vector{Matrix{T}}
	cx_tmp::Vector{Matrix{T}}

    lhs::Vector{Matrix{T}}
    lhs_tl::Vector{SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}}
    lhs_tr::Vector{SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}}
    lhs_bl::Vector{SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}}
    lhs_br::Vector{SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}}

    kkt_matrix_ws::Vector{BunchKaufmanWs{T}}
    D_cache::Vector{Pair{Vector{T}}}
end

function update_rule_parameters(T, constraints::Constraints)
    N = length(constraints)

    eq = [zeros(T, c.num_control + c.num_constraint, c.num_state + 1) for c in constraints]
    ineq = [zeros(T, 2 * c.num_control, c.num_state + 1) for c in constraints]

    α = [@views eq[t][1:constraints[t].num_control, 1] for t in 1:N]
    β = [@views eq[t][1:constraints[t].num_control, 2:end] for t in 1:N]

	ψ = [@views eq[t][constraints[t].num_control+1:end, 1] for t in 1:N]
    ω = [@views eq[t][constraints[t].num_control+1:end, 2:end] for t in 1:N]

    χl = [@views ineq[t][1:constraints[t].num_control, 1] for t in 1:N]
    ζl = [@views ineq[t][1:constraints[t].num_control, 2:end] for t in 1:N]
    
    χu = [@views ineq[t][constraints[t].num_control+1:end, 1] for t in 1:N]
    ζu = [@views ineq[t][constraints[t].num_control+1:end, 2:end] for t in 1:N]

    return UpdateRuleParameters(eq, ineq, α, β, ψ, ω, χl, ζl, χu, ζu)
end

function update_rule_data(T, constraints::Constraints)
    parameters = update_rule_parameters(T, constraints)

    # value function approximation
    V̄xx = [zeros(T, c.num_state, c.num_state) for c in constraints]
    V̄x =  [zeros(T, c.num_state) for c in constraints]
    value = Value(V̄x, V̄xx)

    # control-value function approximation
    Qû = [zeros(T, c.num_control) for c in constraints]
    C = deepcopy(V̄xx)
    Ĥ = [zeros(T, c.num_control, c.num_control) for c in constraints]
    B = [zeros(T, c.num_control, c.num_state) for c in constraints]

    x_tmp = deepcopy(V̄x)
    u_tmp1 = [zeros(T, c.num_control) for c in constraints]
    u_tmp2 = deepcopy(u_tmp1)
	c_tmp = [zeros(T, c.num_constraint) for c in constraints]
	uu_tmp = deepcopy(Ĥ)
	ux_tmp = deepcopy(B)
    xx_tmp = deepcopy(C)

    cu_tmp = [zeros(T, c.num_constraint, c.num_control) for c in constraints]
    cx_tmp = [zeros(T, c.num_constraint, c.num_state) for c in constraints]

    lhs = [zeros(T, c.num_constraint + c.num_control, c.num_constraint + c.num_control) for c in constraints]
    lhs_tl = [@views lhs[t][1:c.num_control, 1:c.num_control] for (t, c) in enumerate(constraints)]
    lhs_tr = [@views lhs[t][1:c.num_control, c.num_control+1:end] for (t, c) in enumerate(constraints)]
    lhs_bl = [@views lhs[t][c.num_control+1:end, 1:c.num_control] for (t, c) in enumerate(constraints)]
    lhs_br = [@views lhs[t][c.num_control+1:end, c.num_control+1:end] for (t, c) in enumerate(constraints)]

    kkt_matrix_ws = [BunchKaufmanWs(L) for L in lhs]
    D_cache = [Pair(zeros(T, c.num_constraint + c.num_control), zeros(T, c.num_constraint + c.num_control)) for c in constraints]

    UpdateRuleData{T}(parameters, value, Qû, C, Ĥ, B,
        x_tmp, u_tmp1, u_tmp2, c_tmp, uu_tmp, ux_tmp, xx_tmp, cu_tmp, cx_tmp,
        lhs, lhs_tl, lhs_tr, lhs_bl, lhs_br, kkt_matrix_ws, D_cache)
end