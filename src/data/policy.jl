""" 
    Value function approximation 
"""
struct Value{N,NN}
    gradient::Vector{N}
    hessian::Vector{NN}
end

""" 
    Action-value function approximation 
"""
struct ActionValue{N,M,NN,MM,MN}
    gradient_state::Vector{N}
    gradient_action::Vector{M}
    hessian_state_state::Vector{NN}
    hessian_action_action::Vector{MM}
    hessian_action_state::Vector{MN}
end

"""
    Store all gains
"""
struct Gains
    kuϕ
    Kuϕ
    ku
    Ku
    kϕ
    Kϕ
    kvl
    Kvl
    kvu
    Kvu
end

"""
    Policy Data
"""
struct PolicyData#{N,M,NN,MM,MN,NNN,MNN,H,HN,HM} 
    # policy u = ū + K * (x - x̄) + k
    gains_main
    gains_soc

    # value function approximation
    value#::Value{N,NN}

    # action-value function approximation
    action_value#::ActionValue{N,M,NN,MM,MN}

    # pre-allocated memory
    x_tmp#::Vector{N}
    u_tmp#::Vector{M}
    h_tmp#::Vector{H}
	uu_tmp#::Vector{MM}
	ux_tmp#::Vector{MN}
    xx_tmp#::Vector{NN}
	hu_tmp#::Vector{HM}
	hx_tmp#::Vector{HN}

    lhs
    lhs_tl
    lhs_tr
    lhs_bl
    lhs_br

    rhs
    rhs_t
    rhs_b

    rhs_x
    rhs_x_t
    rhs_x_b

    lhs_bk
end

function gains_data(dynamics::Vector{Dynamics{T, J}}, constraints::Constraints{T, J}) where {T, J}
    N = length(dynamics) + 1

    Kuϕ = [zeros(dynamics[k].num_control + constraints[k].num_constraint, dynamics[k].num_state)
                for k = 1:N-1]
    kuϕ = [zeros(dynamics[k].num_control + constraints[k].num_constraint) for k = 1:N-1]

	Ku = [@views Kuϕ[k][1:dynamics[k].num_control, :] for k = 1:N-1]
    ku = [@views kuϕ[k][1:dynamics[k].num_control] for k = 1:N-1]

	Kϕ = [@views Kuϕ[k][dynamics[k].num_control+1:end, :] for k = 1:N-1]
    kϕ = [@views kuϕ[k][dynamics[k].num_control+1:end] for k = 1:N-1]

    kvl = [zeros(d.num_control) for d in dynamics]
    Kvl = [zeros(d.num_control, d.num_state) for d in dynamics]

    kvu = [zeros(d.num_control) for d in dynamics]
    Kvu = [zeros(d.num_control, d.num_state) for d in dynamics]
    return Gains(kuϕ, Kuϕ, ku, Ku, kϕ, Kϕ, kvl, Kvl, kvu, Kvu)
end

function policy_data(dynamics::Vector{Dynamics{T, J}}, constraints::Constraints{T, J}) where {T, J}
    H = length(dynamics) + 1

    gains_main = gains_data(dynamics, constraints)
    gains_soc = gains_data(dynamics, constraints)

    # value function approximation
    P = [[zeros(d.num_state, d.num_state) for d in dynamics]..., 
            zeros(dynamics[end].num_next_state, dynamics[end].num_next_state)]
    p =  [[zeros(d.num_state) for d in dynamics]..., 
            zeros(dynamics[end].num_next_state)]
    value = Value(p, P)

    # action-value function approximation
    Qx = [zeros(d.num_state) for d in dynamics]
    Qu = [zeros(d.num_control) for d in dynamics]
    Qxx = [zeros(d.num_state, d.num_state) for d in dynamics]
    Quu = [zeros(d.num_control, d.num_control) for d in dynamics]
    Qux = [zeros(d.num_control, d.num_state) for d in dynamics]

    action_value = ActionValue(Qx, Qu, Qxx, Quu, Qux)

    x_tmp = [[zeros(d.num_state) for d in dynamics]..., zeros(dynamics[end].num_next_state)]
    u_tmp = [zeros(d.num_control) for d in dynamics]
	h_tmp = [zeros(g.num_constraint) for g in constraints]
	uu_tmp = [zeros(d.num_control, d.num_control) for d in dynamics]
	ux_tmp = [zeros(d.num_control, d.num_state) for d in dynamics]
    xx_tmp = [zeros(d.num_state, d.num_state) for d in dynamics]

    hu_tmp = [zeros(constraints[t].num_constraint, dynamics[t].num_control) for t = 1:H-1]
    hx_tmp = [zeros(constraints[t].num_constraint, dynamics[t].num_state) for t = 1:H-1]

    lhs = [zeros(constraints[t].num_constraint + dynamics[t].num_control,
            constraints[t].num_constraint + dynamics[t].num_control) for t = 1:H-1]
    lhs_tl = [@views lhs[t][1:dynamics[t].num_control, 1:dynamics[t].num_control] for t = 1:H-1]
    lhs_tr = [@views lhs[t][1:dynamics[t].num_control, dynamics[t].num_control+1:end] for t = 1:H-1]
    lhs_bl = [@views lhs[t][dynamics[t].num_control+1:end, 1:dynamics[t].num_control] for t = 1:H-1]
    lhs_br = [@views lhs[t][dynamics[t].num_control+1:end, dynamics[t].num_control+1:end] for t = 1:H-1]

    rhs = [zeros(constraints[t].num_constraint + dynamics[t].num_control) for t = 1:H-1]
    rhs_t = [@views rhs[t][1:dynamics[t].num_control] for t = 1:H-1]
    rhs_b = [@views rhs[t][dynamics[t].num_control+1:end] for t = 1:H-1]

    rhs_x = [zeros(constraints[t].num_constraint + dynamics[t].num_control, dynamics[t].num_state) for t = 1:H-1]
    rhs_x_t = [@views rhs_x[t][1:dynamics[t].num_control, :] for t = 1:H-1]
    rhs_x_b = [@views rhs_x[t][dynamics[t].num_control+1:end, :] for t = 1:H-1]

    lhs_bk = [bunchkaufman(L, true; check=false) for L in lhs]

    PolicyData(gains_main, gains_soc,
        value, action_value,
        x_tmp, u_tmp, h_tmp, uu_tmp, ux_tmp, xx_tmp, hu_tmp, hx_tmp,
        lhs, lhs_tl, lhs_tr, lhs_bl, lhs_br, rhs, rhs_t, rhs_b, rhs_x, rhs_x_t, rhs_x_b, lhs_bk)
end