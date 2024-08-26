using Enzyme

x = [2.0, 2.0]
y = Vector{Float64}(undef, 1)

vx = ([1.0, 0.0], [0.0, 1.0])
hess = ([0.0, 0.0], [0.0, 0.0])
dx = [0.0, 0.0]
dy = [1.0]

function f(x::Array{Float64}, y::Array{Float64})
    y[1] = x[1] * x[1] + x[2] * x[1]
    return nothing
end;

function grad(x, dx, y, dy)
    Enzyme.autodiff_deferred(Reverse, f, Duplicated(x, dx), DuplicatedNoNeed(y, dy))
    nothing
  end

using BenchmarkTools
display(@benchmark Enzyme.autodiff(Enzyme.Forward, grad,
                Enzyme.BatchDuplicated(x, vx),
                Enzyme.BatchDuplicated(dx, hess),
                Const(y),
                Const(dy)))ß

display(@benchmark grad(x, dx, y, dy))