struct Bound{T}
    lower::Vector{T}
    upper::Vector{T}
    indices_lower::Vector{Int}
    indices_upper::Vector{Int}
    num_lower::Int
    num_upper::Int
end

Bounds{T} = Vector{Bound{T}} where T

function Bound(lower::Vector{T}, upper::Vector{T}) where T
    indices_lower = [i for (i, mask) in enumerate(.!isinf.(lower)) if mask]
    indices_upper = [i for (i, mask) in enumerate(.!isinf.(upper)) if mask]
    num_lower = sum([1 for b in lower if !isinf(b)])
    num_upper = sum([1 for b in upper if !isinf(b)])
    return Bound{T}(lower, upper, indices_lower, indices_upper, num_lower, num_upper)
end

function Bound(T, num_control::Int)
    return Bound(-T(Inf) .* ones(T, num_control), T(Inf) .* ones(T, num_control))
end

function Bound(num_control::Int, lower::T, upper::T) where T
    return Bound(lower .* ones(T, num_control), upper .* ones(T, num_control))
end
