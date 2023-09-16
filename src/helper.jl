using TensorOperations 

function tensordot_naive(a::Vector, b::Vector)
    result = sum(a .* b, dims=1)
    return permutedims(result, [2, 3, 1])

end 

# TODO: Implement with tensor operations 