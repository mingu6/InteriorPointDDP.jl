using JuMP
import Ipopt
using Random

h = 0.01
N = 101
xN = [1.0; 0.0]
x1 = [0.0; 0.0]

nx = 2  # num. state
nu = 3  # num. control

Random.seed!(0)

function blockmove_continuous(x, u)
    return [x[2], u[1]]
end
blockmove_discrete = (x, u) -> x + h * blockmove_continuous(x + 0.5 * h * blockmove_continuous(x, u), u)

function cost(x, u)
    J = 0.0
    for k = 1:N-1
        J += h * (u[k, 2] + u[k, 3])
    end
    J += 400.0 * (x[N, :] - xN)' * (x[N, :] - xN)
    return J
end

model = Model(
    optimizer_with_attributes(Ipopt.Optimizer, "nlp_scaling_method" => "none")
    );

@variable(model, x[1:N, 1:nx]);
@variable(model, u[1:N-1, 1:nu]);

@objective(model, Min, cost(x, u))

@constraint(model, x[1, :] == x1)
for k = 1:N-1
    @constraint(model, x[k+1, :] == blockmove_discrete(x[k, :], u[k, :]))
    @constraint(model, u[k, 2] - u[k, 3] == u[k, 1] * x[k, 2])
    @constraint(model, -10.0 <= u[k, 1] <= 10.0)
    @constraint(model, u[k, 2:3] .>= 0.0)
end

# initialise variables and solve

ū = [[1.0e-2 * randn(1); 0.01 * ones(2)] for k = 1:N-1]
x̄ = [x1]
for k in 2:N
    push!(x̄, blockmove_discrete(x̄[k-1],  ū[k-1]))
end

for k = 1:N
    for j = 1:nx
        set_start_value(x[k, j], x̄[k][j])
    end
end

for k = 1:N-1
    for j = 1:nu
        set_start_value(u[k, j], ū[k][j])
    end
end

optimize!(model)
