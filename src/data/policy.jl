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
    Policy Data
"""
struct PolicyData{N,M,NN,MM,MN,NNN,MNN,S,SN} # TODO: add S and SN
    # policy u = ū + K * (x - x̄) + k
    Ku::Vector{MN} # β
    ku::Vector{M}  # α
    ## TODO: refactor to Ku and ku

    # S = # of inequality dual variables
    # N = # of states
    Ks::Vector{SN} # θ
    ks::Vector{S} # η

    Ky::Vector{SN} # χ
    ky::Vector{S} # ζ

    # value function approximation
    value::Value{N,NN}

    # action-value function approximation
    action_value::ActionValue{N,M,NN,MM,MN}

    # pre-allocated memory
    x_tmp::Vector{N}
    u_tmp::Vector{M}
	xx̂_tmp::Vector{NNN}
	ux̂_tmp::Vector{MNN}
	uu_tmp::Vector{MM}
	ux_tmp::Vector{MN}
end

function policy_data(dynamics::Vector{Dynamics{T}}, constraints::ConstraintsData) where T
    # policy
	Ku = [zeros(d.num_action, d.num_state) for d in dynamics]
    ku = [zeros(d.num_action) for d in dynamics]

    H = length(dynamics) # number of time steps
	Ks = [zeros(constraints[t].num_inequality, dynamics[t].num_state) for t = 1:H] # S by N
    ks = [zeros(c.num_inequality) for c in constraints] # S-length vector

	Ky = [zeros(constraints[t].num_inequality, dynamics[t].num_state) for t = 1:H] # S by N
    ky = [zeros(c.num_inequality) for c in constraints] # S-length vector

    # value function approximation
    P = [[zeros(d.num_state, d.num_state) for d in dynamics]..., 
            zeros(dynamics[end].num_next_state, dynamics[end].num_next_state)]
    p =  [[zeros(d.num_state) for d in dynamics]..., 
            zeros(dynamics[end].num_next_state)]

    value = Value(p, P)

    # action-value function approximation
    Qx = [zeros(d.num_state) for d in dynamics]
    Qu = [zeros(d.num_action) for d in dynamics]
    Qxx = [zeros(d.num_state, d.num_state) for d in dynamics]
    Quu = [zeros(d.num_action, d.num_action) for d in dynamics]
    Qux = [zeros(d.num_action, d.num_state) for d in dynamics]

    action_value = ActionValue(Qx, Qu, Qxx, Quu, Qux)

    x_tmp = [zeros(d.num_state) for d in dynamics]
    u_tmp = [zeros(d.num_action) for d in dynamics]
	xx̂_tmp = [zeros(d.num_state, d.num_next_state) for d in dynamics]
	ux̂_tmp = [zeros(d.num_action, d.num_next_state) for d in dynamics]
	uu_tmp = [zeros(d.num_action, d.num_action) for d in dynamics]
	ux_tmp = [zeros(d.num_action, d.num_state) for d in dynamics]

    PolicyData(Ku, ku, Ks, ks, Ky, ky,
        value,
        action_value,
        x_tmp, u_tmp, xx̂_tmp, ux̂_tmp, uu_tmp, ux_tmp)
end