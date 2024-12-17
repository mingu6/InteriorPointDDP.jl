struct Constraint
    evaluate
    jacobian_state
    jacobian_control
    vhxx
    vhux  # DDP tensor contraction
    vhuu  # DDP tensor contraction
    num_constraint::Int
    num_state::Int
    num_control::Int
    indices_compl::Vector{Int}
end

Constraints = Vector{Constraint}

function Constraint(f::Function, num_state::Int, num_control::Int; quasi_newton::Bool=false, indices_compl=nothing)

    x = Symbolics.variables(:x, 1:num_state)
    u = Symbolics.variables(:u, 1:num_control)

    evaluate = f(x, u)
    jacobian_state = Symbolics.jacobian(evaluate, x)
    jacobian_control = Symbolics.jacobian(evaluate, u)

    evaluate_func = eval(Symbolics.build_function(evaluate, x, u)[2])
    jacobian_state_func = eval(Symbolics.build_function(jacobian_state, x, u)[2])
    jacobian_control_func = eval(Symbolics.build_function(jacobian_control, x, u)[2])

    num_constraint = length(evaluate)
    
    v = Symbolics.variables(:v, 1:num_constraint)  # vector variables for Hessian vector products

    if !quasi_newton && num_constraint > 0
        vhxx = Symbolics.jacobian(jacobian_state' * v, x)
        vhux = Symbolics.jacobian(jacobian_control' * v, x)
        vhuu = Symbolics.jacobian(jacobian_control' * v, u)
        vhxx_func = eval(Symbolics.build_function(vhxx, x, u, v)[2])
        vhux_func = eval(Symbolics.build_function(vhux, x, u, v)[2])
        vhuu_func = eval(Symbolics.build_function(vhuu, x, u, v)[2])
    else
        vhxx_func = nothing
        vhux_func = nothing
        vhuu_func = nothing
    end

    indices_compl = isnothing(indices_compl) ? Int64[] : indices_compl 

    return Constraint(evaluate_func, jacobian_state_func, jacobian_control_func, vhxx_func, vhux_func, vhuu_func,
        num_constraint, num_state, num_control, indices_compl)
end

function Constraint(num_state::Int, num_control::Int)
    return Constraint(
        (c, x, u) -> nothing,
        (c, x, u) -> nothing, (c, x, u) -> nothing,
        (c, x, u, v) -> nothing, (c, x, u, v) -> nothing, (c, x, u, v) -> nothing,
        0, num_state, num_control, Int64[])
end

function Constraint(h::Function, hx::Function, hu::Function, num_constraint::Int, num_state::Int, num_control::Int;
    indices_compl=nothing, vhxx::Function=nothing, vhux::Function=nothing, vhuu::Function=nothing)
    return Constraint(h, hx, hu, vhxx, vhux, vhuu,
        num_constraint, num_state, num_control, indices_compl)
end

function jacobian!(jacobian_states::Vector{Matrix{T}}, jacobian_controls::Vector{Matrix{T}}, constraints::Constraints,
            states::Vector{Vector{T}}, controls::Vector{Vector{T}}) where T
    for (t, con) in enumerate(constraints)
        con.num_constraint == 0 && continue
        con.jacobian_state(jacobian_states[t], states[t], controls[t])
        con.jacobian_control(jacobian_controls[t], states[t], controls[t])
    end
end

function tensor_contraction!(vhxx_cache::Matrix{T}, vhux_cache::Matrix{T}, vhuu_cache::Matrix{T}, constraint::Constraint,
            x::Vector{T}, u::Vector{T}, v::Vector{T}) where T
    if !isnothing(constraint.vhuu)
        constraint.vhxx(vhxx_cache, x, u, v)
        constraint.vhux(vhux_cache, x, u, v)
        constraint.vhuu(vhuu_cache, x, u, v)
    end
end

function tensor_contraction!(vhxx_cache::Vector{Matrix{T}}, vhux_cache::Vector{Matrix{T}}, vhuu_cache::Vector{Matrix{T}}, constraints::Constraints,
            states::Vector{Vector{T}}, controls::Vector{Vector{T}}, v::Vector{Vector{T}}) where T
    N = length(states)
    for t in 1:N-1
        if !isnothing(constraints[t].vhuu)
            constraints[t].vhxx(vhxx_cache[t], states[t], controls[t], v[t])
            constraints[t].vhux(vhux_cache[t], states[t], controls[t], v[t])
            constraints[t].vhuu(vhuu_cache[t], states[t], controls[t], v[t])
        end
    end
end
