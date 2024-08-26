struct VectorValuedADFunction{T, J}
    evaluate!::Function
    jacobian!::Function
    second_order!::Function
    num_input::Int
    num_output::Int
    evaluate_cache::Vector{T}
    jacobian_cache::J
    second_order_cache::Matrix{T}
end

# TODO: use actual in-place evals of f?
function VectorValuedADFunction(f::Function, num_input::Int, num_output::Int)
    eval_cache = zeros(num_output)
    f_ip! = (res, x) -> res .= f(x)

    res_jac = DiffResults.JacobianResult(zeros(num_output), zeros(num_input))
    cfg_jac = ForwardDiff.JacobianConfig(f, zeros(num_input))
    f_jac! = (res, x) -> ForwardDiff.jacobian!(res, f, x, cfg_jac)

    second_order_cache = zeros(num_input, num_input)
    second_order! = (cache, x, v) -> ForwardDiff.jacobian!(cache, x -> ForwardDiff.jacobian(f, x)' * v, x)

    return VectorValuedADFunction(
        f_ip!, f_jac!, second_order!,
        num_input, num_output,
        eval_cache, res_jac, second_order_cache)
end

function VectorValuedADFunction(f::Function, num_input::Int)
    num_output = length(f(zeros(num_input)))
    return VectorValuedADFunction(f, num_input, num_output)
end

function VectorValuedADFunction()
    return VectorValuedADFunction(
        (c, x, u) -> nothing, (j, x, u) -> nothing, (h, x, u, v) -> nothing,
        0, 0,
        Float64[], Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0))
end

function VectorValuedADFunction(f::Function, fx::Function, num_constraint::Int, T,
                                num_state::Int, num_action::Int, vfxx::Function=nothing)

    return VectorValuedADFunction(
        f, fx, vfxx,
        num_input, num_output,
        zeros(T, num_constraint), zeros(T, num_constraint, num_state + num_action),
        zeros(T, num_state + num_action, num_state + num_action))
end

function evaluate!(res::Vector{T}, ad_fn::VectorValuedADFunction{T, J}, x::Vector{T}) where {T, J}
    ad_fn.evaluate!(res, x)
end

function jacobian!(res::Matrix{T}, ad_fn::VectorValuedADFunction{T, J}, x::Vector{T}) where {T, J}
    ad_fn.num_outputs != 0 && ad_fn.jacobian!(res, x)
end

function second_order_contraction!(res::Matrix{T}, ad_fn::VectorValuedADFunction{T, J},
                                   x::Vector{T}, v::Vector{T}) where {T, J}
    !isnothing(ad_fn.second_order) && ad_fn.second_order!(res, x, v)
end
