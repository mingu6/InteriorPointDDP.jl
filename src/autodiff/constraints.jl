
struct Constraint{T, J}
    fn::VectorValuedADFunction{T, J}
    num_constraint::Int
    num_state::Int
    num_control::Int
    indices_complementarity::Vector{Int}
end

Constraints{T, J} = Vector{Constraint{T, J}} where {T, J}

function Constraint(f::Function, num_constraint::Int, num_state::Int, num_control::Int;
                    indices_complementarity::Union{Vector{Int}, Nothing}=nothing)
    indices_complementarity = isnothing(indices_complementarity) ? Int64[] : indices_complementarity
    f_comb = z -> f(z[1:num_state], z[num_state+1:end])
    ad_fn = VectorValuedADFunction(f_comb, num_state + num_control, num_constraint)
    Constraint(ad_fn, num_constraint, num_state, num_control, indices_complementarity)
end

function Constraint(f::Function, num_state::Int, num_control::Int;
                    indices_complementarity::Union{Vector{Int}, Nothing}=nothing)
    num_constraint = length(f(zeros(num_state), zeros(num_control)))
    return Constraint(f, num_constraint, num_state, num_control; indices_complementarity)
end

function Constraint()
    return Constraint(VectorValuedADFunction(), 0, 0, 0, Int64[])
end

function evaluate!(res::Vector{T}, con::Constraint{T, J}, x::Vector{T}, u::Vector{T}) where {T, J}
    con.fn.evaluate!(res, [x; u])
end

function jacobian!(res::Matrix{T}, con::Constraint{T, J}, x::Vector{T}, u::Vector{T}) where {T, J}
    con.fn.jacobian!(res, [x; u])
end

function second_order_contraction!(res::Matrix{T}, con::Constraint{T, J}, x::Vector{T},
                                   u::Vector{T}, v::Vector{T}) where {T, J}
    con.fn.second_order!(res, [x; u], v)
end

