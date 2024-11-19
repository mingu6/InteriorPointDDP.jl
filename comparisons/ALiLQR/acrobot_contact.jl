using IterativeLQR
using LinearAlgebra
using Random
using Plots
using MeshCat
using BenchmarkTools
using Printf

visualise = true
benchmark = true
verbose = true

h = 0.05
N = 101

options = Options()
options.scaling_penalty = 1.2
options.initial_constraint_penalty = 1e-5
options.max_iterations = 300
options.max_dual_updates = 30
options.objective_tolerance = 1e-7
options.lagrangian_gradient_tolerance = 1e-7

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
x0 = [q1; q2]
qN = [π; 0.0]
xN = [qN; qN]

# ## Dynamics - implicit variational integrator (midpoint)

dyn_acrobot = Dynamics((x, u) -> [x[nq .+ (1:nq)]; u[nu .+ (1:nq)]], nx, ny)
dynamics = [dyn_acrobot for k = 1:N-1]

# ## Costs

function objk(x, u)
	J = 0.0 

	q1 = x[1:acrobot_impact.nq] 
	q2 = x[acrobot_impact.nq .+ (1:acrobot_impact.nq)]
	v1 = (q2 - q1) ./ h

	J += 0.01 * h * transpose(v1) * v1
	J += 0.01 * h * u[1] * u[1]
	return J
end

function objN(x, u)
	J = 0.0 
	
	q1 = x[1:acrobot_impact.nq] 
	q2 = x[acrobot_impact.nq .+ (1:acrobot_impact.nq)] 
	v1 = (q2 - q1) ./ h

	J += 100.0 *  dot(v1, v1)
    J += 500.0 * dot(q2 - qN, q2 - qN)
	return J
end

stage = Cost(objk, nx, ny)
objective = [
    [stage for k = 1:N-1]...,
    Cost(objN, nx, 0),
]

# ## Constraints - perturb complementarity to make easier

stage_constr = Constraint((x, u) -> [
            implicit_contact_dynamics(acrobot_impact, x, u, h, 1e-3);
            -u[nq+2:nq+2+2*nc:end]
            ],
            nx, ny)

constraints = [[stage_constr for k = 1:N-1]...,
                Constraint()
                ]

# ## Initialise solver and solve

solver = Solver(dynamics, objective, constraints; options=options)

open("results/acrobot_contact.txt", "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)  solver (ms)  \n")
	for seed = 1:2
		solver.options.verbose = verbose
		Random.seed!(seed)
		
		q2_init = LinRange(q1, qN, N)[2:end]
		ū = [[1.0e-1 * (rand(nu) .- 0.5); q2_init[k]; 0.01 * ones(nc); 0.01 * ones(nc)] for k = 1:N-1]
		
		x̄ = rollout(dynamics, x0, ū)
		
		solve!(solver, x̄, ū)
		
		if benchmark
            solver.options.verbose = false
            solve_time = @belapsed solve!($solver, $x̄, $ū) samples=10
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e    %5.1f       %5.1f\n", seed, solver.data.iterations[1], solver.data.status[1], solver.data.objective[1],
                            solver.data.max_violation[1], solve_time * 1000, 0.0)
        else
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e \n", seed, solver.data.iterations[1], solver.data.status[1], solver.data.objective[1], solver.data.max_violation[1])
        end
	end
end

# ## Plot solution

if visualise
	x_sol, u_sol = get_trajectory(solver)
	x_mat = reduce(vcat, transpose.(x_sol))
	q1 = x_mat[:, 1]
	q2 = x_mat[:, 2]
	v1 = (x_mat[:, 3] - x_mat[:, 1]) ./ h
	v2 = (x_mat[:, 4] - x_mat[:, 2]) ./ h
	u_mat = [map(x -> x[1], u_sol); 0.0]
	λ1 = [map(x -> x[end-1], u_sol); 0.0]
	λ2 = [map(x -> x[end], u_sol); 0.0]
	plot(range(0, (N-1) * h, length=N), [q2 λ1 λ2 v2], label=["q2" "λ1" "λ2" "v2"])
	savefig("plots/acrobot_contact.png")

	q_sol = state_to_configuration(solver.problem.nominal_states)
	visualize!(vis, acrobot_impact, q_sol, Δt=h);
end
