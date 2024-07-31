using InteriorPointDDP
using LinearAlgebra
using Random
using Printf
using BenchmarkTools
using Plots


struct BenchmarkProblem{T}
    horizon::Int
    dt::T
    num_state::Int
    num_action::Int
    x1::Vector{T}
    dynamics::Vector{Dynamics{T}}
    objective::Vector{Cost{T}}
    constraints::Vector{Constraint{T}}
end

include("cartpole.jl")
include("acrobot.jl")
include("invpend.jl")
include("car.jl")
include("concar.jl")
include("unicycle.jl")
include("arm.jl")
include("blockmove.jl")

function display_solve_results(trial, exper_name, solver, file)
    data = solver.data
    options = solver.options
    @printf(file, "%10s %3d    %5s   %5s   %6s   %.4e   %.4e   %.4e   % 1.2f  %7i      %5.2f \n",
        exper_name, trial, options.feasible, options.quasi_newton, data.status,
        data.dual_inf, data.primal_inf, data.cs_inf,
        log10(data.μ), data.k, data.wall_time * 1000)
    flush(file)
end

function setup_problem(env_name::String; kwargs...)
    if env_name == "cartpole"
        problem = setup_cartpole(; kwargs...)
    elseif env_name == "acrobot"
        problem = setup_acrobot(; kwargs...)
    elseif env_name == "arm"
        problem = setup_arm(; kwargs...)
    elseif env_name == "invpend"
        problem = setup_invpend(; kwargs...)
    elseif env_name == "car"
        problem = setup_car(; kwargs...)
    elseif env_name == "concar"
        problem = setup_concar(; kwargs...)
    elseif env_name == "unicycle"
        problem = setup_unicycle(; kwargs...)
    elseif env_name == "arm"
        problem = setup_arm(; kwargs...)
    elseif env_name == "blockmove"
        problem = setup_blockmove(; kwargs...)
    else
        throw(DomainError("invalid environment name " * env_name))
    end
    return problem
end

function solve_problem(problem::BenchmarkProblem; seed::Int=0, multiplier::Float64=0.02, options=Options())
    N = problem.horizon
    rng = Xoshiro(seed)
    ū = [multiplier .* (rand(rng, problem.constraints[k].num_action) .- 1 / 2) for k = 1:N-1]
    # ū = [[2.0, 0.0, 0.0] for k = 1:N-1] # blockmove
    x̄ = Base.invokelatest(rollout, problem.dynamics, problem.x1, ū)
    solver = Solver(problem.dynamics, problem.objective, problem.constraints, options=options)
    Base.invokelatest(solve!, solver, problem.x1, ū)
    return solver, x̄, ū
end

function benchmark_ipddp_all()
    num_trials = 3
    oc_problems = [
        # ("invpend", "invpend", Dict(), Dict()),
        # ("arm", "arm", Dict(), Dict()),
        # ("car", "car", Dict(), Dict()),
        # ("concar", "concar", Dict(), Dict()),
        # ("unicycle", "unicycle", Dict(), Dict()),
        # ("unicy_up", "unicycle", Dict(:x1 => [-10.0; 0.0; 0.2]), Dict()),
        # ("uni_long", "unicycle", Dict(:x1 => [-10.0; 0.0; 0.0], :horizon => 601), Dict()),
        ("cartpole", "cartpole", Dict(), Dict(:multiplier => 1.0)),
        # ("acrobot", "acrobot", Dict(), Dict(:multiplier => 1.0)),
    ]
    
    open("results.txt", "w") do file
        header = "---------------------------------------------------------------------------------------------------------------\n" *
                 "   exper   trial   feas     qn    status     du_inf       pr_inf       cs_inf       μ    iterations  wall_time \n" *
                 "---------------------------------------------------------------------------------------------------------------\n"
        for (name, env, kwargs_prob, kwargs_sol) in oc_problems
            write(file, header)
            problem = setup_problem(env; kwargs_prob...)
            for qn in [true, false]
                for feasible in [true false]
                    options = Options(feasible=feasible, quasi_newton=qn, reset_cache=true, optimality_tolerance=1e-3)
                    solver, _, _ = solve_problem(problem; options=options, seed=0, kwargs_sol...)  # burn in compilation time
                    for trial in 1:num_trials
                        solver, _, _ = solve_problem(problem; options=options, seed=trial, kwargs_sol...)
                        display_solve_results(trial, name, solver, file)
                    end
                end
            end
        end
        @printf(file, "--------------------------------------------------------------------------------------------------------------\n")
    end
end

function benchmark_ipddp_single(env_name::String; seed::Int=0, multiplier::Float64=0.02, options=Options(verbose=true),
                                plot_soln=false, run_benchmark=false, kwargs...)
    problem = setup_problem(env_name; kwargs...)
    solver, x̄, ū = solve_problem(problem; options=options, seed=seed, multiplier=multiplier)
    
    if plot_soln
        x_sol, u_sol = get_trajectory(solver)
        plot_x = plot(hcat(x_sol...)')
        plot_u = plot(hcat(u_sol...)', linetype=:steppost)
        display(plot(plot_x, plot_u, layout=(1, 2), legend=true))
    end
    
    if run_benchmark
        solver.options.verbose = false
        info = @benchmark solve!($solver, $x̄, $ū)
        display(info)
    end
end
