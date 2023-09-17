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
    K::Vector{MN} # β
    k::Vector{M}  # α
    ## refactor to Ku and ku

    # TODO: modify to include eta and theta for δs
    # Ku::Vector{MN} # β
    # ku::Vector{M} # α

    # S = # of inequality dual variables
    # N = # of states
    Ks::Vector{SN} # θ
    ks::Vector{S} # η

    # TODO: NOT IMPORTANT AND FLAG
    # What is the K_candidate for?
    K_candidate::Vector{MN} 
    k_candidate::Vector{M}

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

function policy_data(dynamics::Vector{Dynamics{T}}) where T
    # policy
	K = [zeros(d.num_action, d.num_state) for d in dynamics]
    k = [zeros(d.num_action) for d in dynamics]

    # TODO: make sure dimensions are number of inequalities 
	Ks = [zeros(d.num_action, d.num_state) for d in dynamics]
    ks = [zeros(d.num_action) for d in dynamics]

    K_candidate = [zeros(d.num_action, d.num_state) for d in dynamics]
    k_candidate = [zeros(d.num_action) for d in dynamics]

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

    PolicyData(K, k, Ks, ks, K_candidate, k_candidate, 
        value,
        action_value,
        x_tmp, u_tmp, xx̂_tmp, ux̂_tmp, uu_tmp, ux_tmp)
end