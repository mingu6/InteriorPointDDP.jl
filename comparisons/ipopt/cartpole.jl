using JuMP
import Ipopt
using Random
using Plots
using MeshCat

h = 0.05
N = 101
visualise = true

Random.seed!(0)

include("../../examples/models/cartpole.jl")

if visualise
	include("../../examples/visualise/visualise_cartpole.jl")
	!@isdefined(vis) && (vis = Visualizer())
	render(vis)
end

nq = cartpole.nq
nu = cartpole.nu
nx = 2 * nq
ny = nu + nq

x1 = [0.0; 0.0; 0.0; 0.0]
xN = [0.0; π; 0.0; 0.0]

model = Model(
    optimizer_with_attributes(Ipopt.Optimizer, "nlp_scaling_method" => "none", "tol" => 1e-7)
    );

# ## Costs

objt = (x, y) -> h * dot(y[1], y[1])
objT = (x, y) -> 400. * dot(x - xN, x - xN)

cost = (x, u) -> begin
	J = 0.0
	for k in 1:N-1
		J += objt(x[k, :], u[k, :])
	end
	J += objT(x[N, :], 0.0)
end

# ## Dynamics - implicit dynamics with RK2 integration

f = (x, y) -> [x[nq .+ (1:nq)]; y[nu .+ (1:nq)]]
cartpole_discrete = (x, y) -> x + h * f(x + 0.5 * h * f(x, y), y)  # Explicit midpoint
dyn_con = (x, y) -> implicit_dynamics(cartpole, x, y) * h

# ## Constraints

@variable(model, x[1:N, 1:nx]);
@variable(model, y[1:N-1, 1:ny]);

@objective(model, Min, cost(x, y))

@constraint(model, x[1, :] == x1)
for k = 1:N-1
    @constraint(model, x[k+1, :] == cartpole_discrete(x[k, :], y[k, :]))
    @constraint(model, dyn_con(x[k, :], y[k, :]) .== 0.0)
    # @constraint(model, 4.0 .>= y[k, 1:nu] .>= -4.0)
end

# ## Initialise solver and solve

ȳ = [1.0e-2 * (rand(ny) .- 0.5) for k = 1:N-1]

x̄ = [x1]
for k in 2:N
    push!(x̄, cartpole_discrete(x̄[k-1],  ȳ[k-1]))
end

for k = 1:N
    for j = 1:nx
        set_start_value(x[k, j], x̄[k][j])
    end
end

for k = 1:N-1
    for j = 1:ny
        set_start_value(y[k, j], ȳ[k][j])
    end
end

optimize!(model)

xv = value.(x)
x_sol = [xv[k, :] for k in 1:N]
yv = value.(y)
u_sol = [yv[k, 1:nu] for k in 1:N-1]

if visualise
	q_sol = [x[1:nq] for x in x_sol]
	visualize!(vis, cartpole, q_sol, Δt=h);
end

