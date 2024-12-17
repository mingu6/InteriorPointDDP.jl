using InteriorPointDDP
using LinearAlgebra
using Random
using Plots
using MeshCat
using Printf
using LaTeXStrings

visualise = false
benchmark = true
verbose = true
quasi_newton = false
n_benchmark = 10

T = Float64
h = 0.05
N = 101

include("models/acrobot.jl")

if visualise
	include("visualise/visualise_acrobot.jl")
	!@isdefined(vis) && (vis = Visualizer())
	render(vis)
end

nq = acrobot_impact.nq
nc = acrobot_impact.nc
nF = acrobot_impact.nu
nx = 2 * nq
nu = nF + nq + 2 * nc

q1 = T[0.0; 0.0]
q2 = T[0.0; 0.0]
x1 = T[q1; q2]
qN = T[π; 0.0]
xN = T[qN; qN]

options = Options{T}(quasi_newton=quasi_newton, verbose=true, κ_μ=0.6, θ_μ=1.2)
		
# ## Dynamics - implicit variational integrator (midpoint)

dyn_acrobot = Dynamics((x, u) -> [x[nq .+ (1:nq)]; u[nF .+ (1:nq)]], nx, nu)
dynamics = [dyn_acrobot for k = 1:N-1]

# ## Costs

function objt(x, u)
	J = 0.01 * h * u[1] * u[1]
	return J
end

function objT(x, u)
	J = 0.0 
	
	q1 = x[1:acrobot_impact.nq] 
	q2 = x[acrobot_impact.nq .+ (1:acrobot_impact.nq)] 
	v1 = (q2 - q1) ./ h

	J += 200.0 *  dot(v1, v1)
	J += 500.0 * dot(q2 - qN, q2 - qN)
	return J
end

stage = Cost(objt, nx, nu)
objective = [
	[stage for k = 1:N-1]...,
	Cost(objT, nx, 0),
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

fname = quasi_newton ? "examples/results/acrobot_contact_QN.txt" : "examples/results/acrobot_contact.txt"
open(fname, "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (s)   solver(s)  \n")
	for seed = 1:50
		solver.options.verbose = verbose
		Random.seed!(seed)
		
		# ## Initialise solver and solve
		
		q2_init = LinRange(q1, qN, N)[2:end]
		ū = [[T(1.0e-2) * (rand(T, nF) .- 0.5); q2_init[k]; T(0.01) * ones(T, nc); T(0.01) * ones(T, nc)] for k = 1:N-1]
		solve!(solver, x1, ū)
				
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
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e    %5.1f          %5.1f  \n", seed, solver.data.k, solver.data.status == 0, solver.data.objective, solver.data.primal_inf, wall_time * 1000, solver_time * 1000)
        else
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e \n", seed, solver.data.k, solver.data.status == 0, solver.data.objective, solver.data.primal_inf)
        end
	end
end

# ## Plot solution

if visualise
	x_sol, u_sol = get_trajectory(solver)
	q_e = map(u -> u[nF + 2], u_sol)
	ϕ1 = map(qe -> π / 2 - qe, q_e)
	ϕ2 = map(qe -> qe + π / 2, q_e)
	λ1 = map(x -> x[end-1], u_sol)
	λ2 = map(x -> x[end], u_sol)
	u = map(x -> x[1], u_sol)
	plot(range(0, h * (N-1), N-1), [ϕ1 ϕ2 λ1 λ2], xtickfontsize=14, ytickfontsize=14, xlabel=L"$t$", ylims=(0,5),
		legendfontsize=12, linewidth=2, linestyle=[:solid :solid :dot :dot], linecolor=[1 2 1 2], 
		background_color_legend = nothing, label=[L"$\phi(q)_1$" L"$\phi(q)_2$" L"$\lambda_1$" L"$\lambda_2$"])
	savefig("examples/plots/acrobot_contact_IPDDP.pdf")
	
	q_sol = state_to_configuration(x_sol)
	visualize!(vis, acrobot_impact, q_sol, Δt=h);
end
