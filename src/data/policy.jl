""" 
    Value function approximation 
"""
struct Value{N,NN}
    gradient::Vector{N}
    hessian::Vector{NN}
end

""" 
    control-value function approximation 
"""
struct Hamiltonian{N,M,NN,MM,MN}
    gradient_state::Vector{N}
    gradient_control::Vector{M}
    hessian_state_state::Vector{NN}
    hessian_control_control::Vector{MM}
    hessian_control_state::Vector{MN}
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
struct PolicyData{T}
    # policy u = ū + K * (x - x̄) + k
    gains_main
    gains_soc

    # value function approximation
    value#::Value{N,NN}

    # control-value (Q) function approximation
    hamiltonian#::controlValue{N,M,NN,MM,MN}

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

function gains_data(T, dynamics::Model, constraints::Constraints)
    N = length(dynamics) + 1

    Kuϕ = [zeros(T, dynamics[t].num_control + constraints[t].num_constraint, dynamics[t].num_state)
                for t = 1:N-1]
    kuϕ = [zeros(T, dynamics[t].num_control + constraints[t].num_constraint) for t = 1:N-1]

	Ku = [@views Kuϕ[t][1:dynamics[t].num_control, :] for t = 1:N-1]
    ku = [@views kuϕ[t][1:dynamics[t].num_control] for t = 1:N-1]

	Kϕ = [@views Kuϕ[t][dynamics[t].num_control+1:end, :] for t = 1:N-1]
    kϕ = [@views kuϕ[t][dynamics[t].num_control+1:end] for t = 1:N-1]

    kvl = [zeros(T, d.num_control) for d in dynamics]
    Kvl = [zeros(T, d.num_control, d.num_state) for d in dynamics]

    kvu = [zeros(T, d.num_control) for d in dynamics]
    Kvu = [zeros(T, d.num_control, d.num_state) for d in dynamics]
    return Gains(kuϕ, Kuϕ, ku, Ku, kϕ, Kϕ, kvl, Kvl, kvu, Kvu)
end

function policy_data(T, dynamics::Vector{Dynamics}, constraints::Constraints)
    N = length(dynamics) + 1

    gains_main = gains_data(T, dynamics, constraints)
    gains_soc = gains_data(T, dynamics, constraints)

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
	h_tmp = [zeros(T, g.num_constraint) for g in constraints]
	uu_tmp = [zeros(T, d.num_control, d.num_control) for d in dynamics]
	ux_tmp = [zeros(T, d.num_control, d.num_state) for d in dynamics]
    xx_tmp = [zeros(T, d.num_state, d.num_state) for d in dynamics]

    hu_tmp = [zeros(T, constraints[t].num_constraint, dynamics[t].num_control) for t = 1:N-1]
    hx_tmp = [zeros(T, constraints[t].num_constraint, dynamics[t].num_state) for t = 1:N-1]

    lhs = [zeros(T, constraints[t].num_constraint + dynamics[t].num_control,
            constraints[t].num_constraint + dynamics[t].num_control) for t = 1:N-1]
    lhs_tl = [@views lhs[t][1:dynamics[t].num_control, 1:dynamics[t].num_control] for t = 1:N-1]
    lhs_tr = [@views lhs[t][1:dynamics[t].num_control, dynamics[t].num_control+1:end] for t = 1:N-1]
    lhs_bl = [@views lhs[t][dynamics[t].num_control+1:end, 1:dynamics[t].num_control] for t = 1:N-1]
    lhs_br = [@views lhs[t][dynamics[t].num_control+1:end, dynamics[t].num_control+1:end] for t = 1:N-1]

    rhs = [zeros(T, constraints[t].num_constraint + dynamics[t].num_control) for t = 1:N-1]
    rhs_t = [@views rhs[t][1:dynamics[t].num_control] for t = 1:N-1]
    rhs_b = [@views rhs[t][dynamics[t].num_control+1:end] for t = 1:N-1]

    rhs_x = [zeros(T, constraints[t].num_constraint + dynamics[t].num_control, dynamics[t].num_state) for t = 1:N-1]
    rhs_x_t = [@views rhs_x[t][1:dynamics[t].num_control, :] for t = 1:N-1]
    rhs_x_b = [@views rhs_x[t][dynamics[t].num_control+1:end, :] for t = 1:N-1]

    lhs_bk = [bunchkaufman(L, true; check=false) for L in lhs]

    PolicyData{T}(gains_main, gains_soc,
        value, hamiltonian,
        x_tmp, u_tmp, h_tmp, uu_tmp, ux_tmp, xx_tmp, hu_tmp, hx_tmp,
        lhs, lhs_tl, lhs_tr, lhs_bl, lhs_br, rhs, rhs_t, rhs_b, rhs_x, rhs_x_t, rhs_x_b, lhs_bk)
end