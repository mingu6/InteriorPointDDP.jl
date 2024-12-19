using InteriorPointDDP
using LinearAlgebra
using Random
using Plots
using MeshCat
using Printf
using LaTeXStrings

visualise = false
benchmark = false
verbose = true
quasi_newton = false
n_benchmark = 10

T = Float64
h = 0.05
N = 101

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

x1 = T[0.0; 0.0; 0.0; 0.0]
qN = T[π; 0.0]
xN = T[qN; qN]

options = Options{T}(quasi_newton=quasi_newton, verbose=true, κ_μ=0.6, θ_μ=1.2)
		
# ## Dynamics - implicit variational integrator (midpoint)

dyn_acrobot = Dynamics((x, u) -> [x[nq .+ (1:nq)]; u[nτ .+ (1:nq)]], nx, nu)
dynamics = [dyn_acrobot for k = 1:N-1]

# ## Costs

function objt(x, u)
	τ = u[1]
	J = 0.01 * h * τ * τ
	return J
end

function objN(x, u)
	J = 0.0 
	
	q⁻ = x[1:acrobot_impact.nq] 
	q = x[acrobot_impact.nq .+ (1:acrobot_impact.nq)] 
	q̇ᵐ⁻ = (q - q⁻) ./ h

	J += 200.0 * dot(q̇ᵐ⁻, q̇ᵐ⁻)
	J += 500.0 * dot(q - qN, q - qN)
	return J
end

stage = Cost(objt, nx, nu)
objective = [
	[stage for k = 1:N-1]...,
	Cost(objN, nx, 0),
]

# ## Constraints

stage_constr = Constraint((x, u) -> implicit_contact_dynamics(acrobot_impact, x, u, h),
			nx, nu, indices_compl=[5, 6])

constraints = [stage_constr for k = 1:N-1]

# ## Bounds

bound = Bound(
	[-T(10.0); -T(Inf) * ones(T, nq); zeros(T, nc); zeros(T, nc)],
	[T(10.0); T(Inf) * ones(T, nq); T(Inf) * ones(T, nc); T(Inf) * ones(T, nc)]
)
bounds = [bound for k in 1:N-1]

solver = Solver(T, dynamics, objective, constraints, bounds, options=options)

fname = quasi_newton ? "results/acrobot_contact_QN.txt" : "results/acrobot_contact.txt"
open(fname, "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)   solver(ms)  \n")
	for seed = 1:50
		solver.options.verbose = verbose
		Random.seed!(seed)
		
		# ## Initialise solver and solve
		
		q_init = LinRange([0.0; 0.0], qN, N)[2:end]
		ū = [[T(1.0e-2) * (rand(T, nτ) .- 0.5); q_init[k]; T(0.01) * ones(T, nc); T(0.01) * ones(T, nc)] for k = 1:N-1]
		state_diffs = solve!(solver, x1, ū)
		
		if solver.data.status == 0
            plot!(1:solver.data.k+1, state_diffs, yaxis=:log10, yticks=[1e2, 1e0, 1e-2, 1e-4, 1e-6, 1e-8],
                    ylims=(1e-9, 3e2), legend=false, linecolor=1, xtickfontsize=14, ytickfontsize=14)
		end

		if benchmark
            solver.options.verbose = false
            solver_time = 0.0
            wall_time = 0.0
            for i in 1:n_benchmark
                solve!(solver, x1, ū)
                solver_time += solver.data.solver_time
                wall_time += solver.data.wall_time
            end
            solver_time /= n_benchmark
            wall_time /= n_benchmark
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e     %5.1f           %5.1f  \n", seed, solver.data.k, solver.data.status == 0, solver.data.objective, solver.data.primal_inf, wall_time * 1000, solver_time * 1000)
        else
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e \n", seed, solver.data.k, solver.data.status == 0, solver.data.objective, solver.data.primal_inf)
        end
	end
end

savefig("plots/acrobot_convergence.pdf")

# # ## Plot solution

# x_sol, u_sol = get_trajectory(solver)
# θe = map(x -> x[4], x_sol[1:end-1])
# s1 = map(θ -> π / 2 - θ, θe)
# s2 = map(θ -> θ + π / 2, θe)
# λ1 = map(u -> u[4], u_sol)
# λ2 = map(u -> u[5], u_sol)
# plot(range(0, h * (N-1), N-1), [s1 s2 λ1 λ2], xtickfontsize=14, ytickfontsize=14, xlabel=L"$t$", ylims=(0,6),
# 	legendfontsize=12, linewidth=2, xlabelfontsize=14, linestyle=[:solid :solid :dot :dot], linecolor=[1 2 1 2], 
# 	background_color_legend = nothing, label=[L"$s_t^{(1)}$" L"$s_t^{(2)}$" L"$\lambda^{(1)}_t$" L"$\lambda^{(2)}_t$"])
# savefig("plots/acrobot_contact_IPDDP.pdf")

# ## Visualise trajectory using MeshCat

if visualise
	q_sol = state_to_configuration(x_sol)
	visualize!(vis, acrobot_impact, q_sol, Δt=h);
end
