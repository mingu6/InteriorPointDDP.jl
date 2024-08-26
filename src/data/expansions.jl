const ColonSlice = Base.Slice{Base.OneTo{Int}}

"""
    DynConExpansion

Stores the expansion of the dynamics/constraints about a point. Stores the jacobians for 
the state/control and stores DDP second-order tensor contractions. Access Jacobians via 
`D.fx` and `D.fu` and tensor contractions using `D.vxx``, `D.vxu`, `D.vuu`.

# Constructor

    DynConExpansion{T}(n, e, m)

where `n` is the input state dimension, `o` is the output dimension
(next state or num. constraints), and `m` is the control dimension.
"""
struct DynConExpansion{T}
    f::Vector{T}   # (No,)
    ∇f::Matrix{T}  # (No, Nx+Nu)
    v∇²f::Matrix{T}  # (Nx+Nu, Nx+Nu) DDP tensor contraction
    fx::SubArray{T,2,Matrix{T},Tuple{ColonSlice, UnitRange{Int}}, true}
    fu::SubArray{T,2,Matrix{T},Tuple{ColonSlice, UnitRange{Int}}, true}
    vxx::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int}, UnitRange{Int}}, false}
    vxu::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int}, UnitRange{Int}}, false}
    vuu::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int}, UnitRange{Int}}, false}
end

function DynConExpansion(T, n::Int, o::Int, m::Int)
    f = zeros(T,o)
    ∇f = zeros(T,o,n+m)
    v∇²f = zeros(T, n+m,n+m)
    fx = view(∇f, :, 1:n)
    fu = view(∇f, :, n+1:n+m)
    vxx = view(v∇²f, 1:n, 1:n)
    vxu = view(v∇²f, 1:n, n+1:n+m)
    vuu = view(v∇²f, n+1:n+m, n+1:n+m)
    return DynConExpansion{T}(f,∇f,v∇²f,fx,fu,vxx,vxu,vuu)
end

function DynConExpansion(d::Dynamics{T}) where T
    return DynConExpansion(T, d.num_state, d.num_next_state, d.num_control)
end

function DynConExpansion(d::Constraint{T}) where T
    return DynConExpansion(T, d.num_state, d.num_constraint, d.num_control)
end

function evaluate!(dyncon::DynConExpansion{T}, fn::Union{Dynamics{T, J}, Constraint{T, J}},
                   x::Vector{T}, u::Vector{T}) where {T, J}
    evaluate!(dyncon.f, fn, x, u)
end

function jacobian!(dyncon::DynConExpansion{T}, fn::Union{Dynamics{T, J}, Constraint{T, J}},
                   x::Vector{T}, u::Vector{T}) where {T, J}
    jacobian!(dyncon.∇f, fn, x, u)
end

function second_order_contraction!(dyncon::DynConExpansion{T}, fn::Union{Dynamics{T, J}, Constraint{T, J}},
                                   x::Vector{T}, u::Vector{T}, v::Vector{T}) where {T, J}
    second_order_contraction!(dyncon.v∇²f, fn, x, u, v)
end

function reset!(dyncon::DynConExpansion{T}) where T 
    fill!(dyncon.f, 0.0)
    fill!(dyncon.∇f, 0.0)
    fill!(dyncon.v∇²f, 0.0)
   
end

function reset!(dyncons::Vector{DynConExpansion{T}}) where T
    for d in dyncons
        reset!(d)
    end
end
