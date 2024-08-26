struct Dynamics{T, J}
    fn::VectorValuedADFunction{T, J}
    num_next_state::Int
    num_state::Int
    num_control::Int
end

Model{T, J} = Vector{Dynamics{T, J}} where {T, J}

function Dynamics(f::Function, num_next_state::Int, num_state::Int, num_control::Int)
    f_comb = z -> f(z[1:num_state], z[num_state+1:end])
    ad_fn = VectorValuedADFunction(f_comb, num_state + num_control, num_next_state)
    Dynamics(ad_fn, num_next_state, num_state, num_control)
end

function Dynamics(f::Function, num_state::Int, num_control::Int)
    num_next_state = length(f(zeros(num_state), zeros(num_control)))
    return Dynamics(f, num_next_state, num_state, num_control)
end

function Dynamics()
    return Dynamics(VectorValuedADFunction(), 0, 0, 0)
end

function evaluate!(res::Vector{T}, dyn::Dynamics{T, J}, x::Vector{T}, u::Vector{T}) where {T, J}
    dyn.fn.evaluate!(res, [x; u])
end

function jacobian!(res::Matrix{T}, dyn::Dynamics{T, J}, x::Vector{T}, u::Vector{T}) where {T, J}
    dyn.fn.jacobian!(res, [x; u])
end

function second_order_contraction!(res::Matrix{T}, dyn::Dynamics{T, J}, x::Vector{T},
                                   u::Vector{T}, v::Vector{T}) where {T, J}
    dyn.fn.second_order!(res, [x; u], v)
end

# TODO: required?
num_trajectory(dynamics::Vector{Dynamics{T, J}}) where {T, J} = sum([d.num_state + d.num_control for d in dynamics]) + dynamics[end].num_next_state
