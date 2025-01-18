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

Δ = 0.05
N = 101

options = Options()
options.max_dual_updates = 1

include("../models/acrobot.jl")

if visualise
	include("../visualise/visualise_acrobot.jl")
	!@isdefined(vis) && (vis = Visualizer())
	render(vis)
end

nq = acrobot_impact.nq
nc = acrobot_impact.nc
nτ = acrobot_impact.nu
nx = 2 * nq
nu = nτ + nq + 2 * nc

x1 = [0.0; 0.0; 0.0; 0.0]
qN = [π; 0.0]
xN = [qN; qN]

# ## Dynamics - implicit variational integrator (midpoint)

dyn_acrobot = Dynamics((x, u) -> [x[nq .+ (1:nq)]; u[nτ .+ (1:nq)]], nx, nu)
dynamics = [dyn_acrobot for k = 1:N-1]

# ## Costs

function objt(x, u)
	τ = u[1]
	J = 0.01 * Δ * τ * τ
	return J
end

function objN(x, u)
	J = 0.0 
	
	q⁻ = x[1:acrobot_impact.nq] 
	q = x[acrobot_impact.nq .+ (1:acrobot_impact.nq)] 
	q̇ᵐ⁻ = (q - q⁻) ./ Δ

	J += 200.0 * dot(q̇ᵐ⁻, q̇ᵐ⁻)
	J += 500.0 * dot(q - qN, q - qN)
	return J
end

stage = Cost(objt, nx, nu)
objective = [
    [stage for k = 1:N-1]...,
    Cost(objN, nx, 0),
]

# ## Constraints - perturb complementarity to make easier

stage_constr = Constraint((x, u) -> [
			implicit_contact_dynamics(acrobot_impact, x, u, Δ, 1e-2);
            u[1] - 10.0;
            -u[1] - 10.0;
            -u[nτ + nq + 1:end]
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
		q_init = LinRange([0.0; 0.0], qN, N)[2:end]
		ū = [[1.0e-2 * (rand(nτ) .- 0.5); q_init[k]; 0.01 * ones(nc); 0.01 * ones(nc)] for k = 1:N-1]
		x̄ = rollout(dynamics, x1, ū)
		
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

Random.seed!(1)
q_init = LinRange([0.0; 0.0], qN, N)[2:end]
ū = [[1.0e-2 * (rand(nτ) .- 0.5); q_init[k]; 0.01 * ones(nc); 0.01 * ones(nc)] for k = 1:N-1]
x̄ = rollout(dynamics, x1, ū)
solve!(solver, x̄, ū)

x_sol, u_sol = get_trajectory(solver)
θe = map(x -> x[4], x_sol[1:end-1])
s1 = map(θ -> π / 2 - θ, θe)
s2 = map(θ -> θ + π / 2, θe)
λ1 = map(u -> u[4], u_sol)
λ2 = map(u -> u[5], u_sol)
plot(range(0, Δ * (N-1), N-1), [s1 s2 λ1 λ2], xtickfontsize=14, ytickfontsize=14, xlabel=L"$t$", ylims=(0,6),
	legendfontsize=12, linewidth=2, xlabelfontsize=14, linestyle=[:solid :solid :dot :dot], linecolor=[1 2 1 2], 
	background_color_legend = nothing, label=[L"$s_t^{(1)}$" L"$s_t^{(2)}$" L"$\lambda^{(1)}_t$" L"$\lambda^{(2)}_t$"])
savefig("plots/acrobot_contact_ALiLQR.pdf")
	
# ## Visualise trajectory using MeshCat

if visualise
	q_sol = state_to_configuration(x_sol)
	visualize!(vis, acrobot_impact, q_sol, Δt=Δ);
end
