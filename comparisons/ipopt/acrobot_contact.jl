using JuMP
import Ipopt
using Random
using Plots
using MeshCat

h = 0.05
N = 101
visualise = true

Random.seed!(0)

include("../../examples/models/acrobot.jl")

if visualise
	include("../../examples/visualise/visualise_acrobot.jl")
	!@isdefined(vis) && (vis = Visualizer())
	render(vis)
end

nq = acrobot_impact.nq
nc = acrobot_impact.nc
nu = acrobot_impact.nu
nx = 2 * nq
ny = nu + nq + 2 * nc

q1 = [0.0; 0.0]
q2 = [0.0; 0.0]
x1 = [q1; q2]
qN = [π; 0.0]
xN = [qN; qN]

model = Model(
    optimizer_with_attributes(Ipopt.Optimizer, "nlp_scaling_method" => "none", "tol" => 1e-8)
    );

acrobot_discrete = (x, u) -> [x[nq .+ (1:nq)]; u[nu .+ (1:nq)]]

# ## Costs

function objt(x, u)
	J = 0.0 

	q1 = x[1:acrobot_impact.nq] 
	q2 = x[acrobot_impact.nq .+ (1:acrobot_impact.nq)]
	v1 = (q2 - q1) ./ h

	J += 0.01 * h * transpose(v1) * v1
	J += 0.01 * h * u[1] * u[1]
	return J
end

function objT(x, u)
	J = 0.0 
	
	q1 = x[1:acrobot_impact.nq] 
	q2 = x[acrobot_impact.nq .+ (1:acrobot_impact.nq)] 
	v1 = (q2 - q1) ./ h

	J += 100.0 *  v1' * v1
    J += 500.0 * (q2 - qN)' * (q2 - qN)
	return J
end

cost = (x, u) -> begin
	J = 0.0
	for k in 1:N-1
		J += objt(x[k, :], u[k, :])
	end
	J += objT(x[N, :], 0.0)
end

# ## Constraints

constr = (x, u) -> implicit_contact_dynamics(acrobot_impact, x, u, h, 1e-6)

@variable(model, x[1:N, 1:nx]);
@variable(model, u[1:N-1, 1:ny]);

@objective(model, Min, cost(x, u))

@constraint(model, x[1, :] == x1)
for k = 1:N-1
    @constraint(model, x[k+1, :] == acrobot_discrete(x[k, :], u[k, :]))
    @constraint(model, constr(x[k, :], u[k, :]) .== 0.0)
    @constraint(model, u[k, (nu + nq) .+ (1:2*nc)] .>= 0.0)
end

q2_init = LinRange(q1, qN, N)[2:end]
ū = [[1.0e-3 * randn(nu); q2_init[k]; 0.01 * ones(nc); 0.01 * ones(nc)] for k = 1:N-1]

x̄ = [x1]
for k in 2:N
    push!(x̄, acrobot_discrete(x̄[k-1],  ū[k-1]))
end

for k = 1:N
    for j = 1:nx
        set_start_value(x[k, j], x̄[k][j])
    end
end

for k = 1:N-1
    for j = 1:ny
        set_start_value(u[k, j], ū[k][j])
    end
end

optimize!(model)

xv = value.(x)
x_sol = [xv[k, :] for k in 1:N]
uv = value.(u)
u_sol = [uv[k, :] for k in 1:N-1]

x_mat = reduce(vcat, transpose.(x_sol))
q1 = x_mat[:, 1]
q2 = x_mat[:, 2]
v1 = (x_mat[:, 3] - x_mat[:, 1]) ./ h
v2 = (x_mat[:, 4] - x_mat[:, 2]) ./ h
u_mat = [map(x -> x[1], u_sol); 0.0]
λ1 = [map(x -> x[end-1], u_sol); 0.0]
λ2 = [map(x -> x[end], u_sol); 0.0]
plot(range(0, (N-1) * h, length=N), [q2 λ1 λ2 v2], label=["q2" "λ1" "λ2" "v2"])
savefig("plots/acrobot_impact.png")

if visualise
	q_sol = state_to_configuration(x_sol)
	visualize!(vis, acrobot_impact, q_sol, Δt=h);
end
