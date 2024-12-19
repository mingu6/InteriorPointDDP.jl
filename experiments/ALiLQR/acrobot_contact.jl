using IterativeLQR
using LinearAlgebra
using Random
using Plots
using MeshCat
using BenchmarkTools
using Printf
using LaTeXStrings

visualise = false
benchmark = false
verbose = true

h = 0.05
N = 101

options = Options()
options.scaling_penalty = 1.0
options.initial_constraint_penalty = 1.0
options.max_iterations = 100
options.max_dual_updates = 50

include("../models/acrobot.jl")

if visualise
	include("../visualise/visualise_acrobot.jl")
	!@isdefined(vis) && (vis = Visualizer())
	render(vis)
end

nq = acrobot_impact.nq
nc = acrobot_impact.nc
nF = acrobot_impact.nu
nx = 2 * nq
nu = nF + nq + 2 * nc

q1 = [0.0; 0.0]
q2 = [0.0; 0.0]
x0 = [q1; q2]
qN = [π; 0.0]
xN = [qN; qN]

# ## Dynamics - implicit variational integrator (midpoint)

dyn_acrobot = Dynamics((x, u) -> [x[nq .+ (1:nq)]; u[nF .+ (1:nq)]], nx, nu)
dynamics = [dyn_acrobot for k = 1:N-1]

# ## Costs

function objk(x, u)
	J = 0.01 * h * u[1] * u[1]
	return J
end

function objN(x, u)
	J = 0.0 
	
	q1 = x[1:acrobot_impact.nq] 
	q2 = x[acrobot_impact.nq .+ (1:acrobot_impact.nq)] 
	v1 = (q2 - q1) ./ h

	J += 200.0 *  dot(v1, v1)
    J += 500.0 * dot(q2 - qN, q2 - qN)
	return J
end

stage = Cost(objk, nx, nu)
objective = [
    [stage for k = 1:N-1]...,
    Cost(objN, nx, 0),
]

# ## Constraints - perturb complementarity to make easier

stage_constr = Constraint((x, u) -> [
            implicit_contact_dynamics(acrobot_impact, x, u, h, 1e-4);
            u[1] - 10.0;
            -u[1] - 10.0;
            -u[nq+2:nq+2+2*nc:end]
            ],
            nx, nu)

constraints = [[stage_constr for k = 1:N-1]...,
                Constraint()
                ]
                
# ## Initialise solver

solver = Solver(dynamics, objective, constraints; options=options)

open("results/acrobot_contact.txt", "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)  solver (ms)  \n")
	for seed = 1:50
		solver = Solver(dynamics, objective, constraints; options=options)
		solver.options.verbose = verbose
		
		Random.seed!(seed)
		q2_init = LinRange(q1, qN, N)[2:end]
		ū = [[1.0e-2 * (rand(nF) .- 0.5); q2_init[k]; 0.01 * ones(nc); 0.01 * ones(nc)] for k = 1:N-1]
		x̄ = rollout(dynamics, x0, ū)
		
		solve!(solver, x̄, ū)
		
		if benchmark
            solver.options.verbose = false
            solve_time = @belapsed solve!(solver, $x̄, $ū) samples=10 setup=(solver=Solver(dynamics, objective, constraints; options=options))
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
	q_e = map(u -> u[nu + 2], u_sol)
	ϕ1 = map(qe -> π / 2 - qe, q_e)
	ϕ2 = map(qe -> qe + π / 2, q_e)
	λ1 = map(x -> x[end-1], u_sol)
	λ2 = map(x -> x[end], u_sol)
	u = map(x -> x[1], u_sol)
	plot(range(0, h * (N-1), N-1), [ϕ1 ϕ2 λ1 λ2], xtickfontsize=14, ytickfontsize=14, xlabel=L"$t$", ylims=(0,5),
		legendfontsize=12, linewidth=2, linestyle=[:solid :solid :dot :dot], linecolor=[1 2 1 2], 
		background_color_legend = nothing, label=[L"$\phi(q)_1$" L"$\phi(q)_2$" L"$\lambda_1$" L"$\lambda_2$"])
	savefig("plots/acrobot_contact_ALiLQR.pdf")
	
	q_sol = state_to_configuration(x_sol)
	visualize!(vis, acrobot_impact, q_sol, Δt=h);
end
