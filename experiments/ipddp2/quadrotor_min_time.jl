using InteriorPointDDP
using LinearAlgebra
using Random
using Plots
using MeshCat
using Printf

visualise = false
benchmark = false
verbose = true
n_benchmark = 10

NPWP = 40
n_ocp = 1

include("../models/quadrotor.jl")

options = Options{Float64}(verbose=verbose, μ_init=0.1, κ_ϵ=100.0, max_iterations=2000)

results = Vector{Vector{Any}}()

for seed = 1:n_ocp
	Random.seed!(seed)

    # standard quad config
    l = 0.15
    Tmin = 0.25
    Tmax = 5.0
    cτ = 0.05
    J = [0.001; 0.001; 0.0017]
    Jinv = [1.0 / J[1]; 1.0 / J[2]; 1.0 / J[3]]
    ω_max_xy = 15.0
    ω_max_z = 0.3
    zmin = 0.5
    zmax = 100.0
    m = 0.85

    vg = 3.0  # magnitude of velocity guess

    wps = [
        [-1.1; -1.6; 3.6],
        [9.2; 6.6; 1.0],
        [9.2; -4.0; 1.2]
    ]
    p1 = [-5; 4.5; 1.2]
    q1 = [1.0; 0.0; 0.0; 0.0]

    nwp = length(wps)

    dtol = 0.3

    nwp = length(wps)
    nx = 13 + nwp + 1
    nu = 4 + 13 + 5 * nwp
    N = nwp * NPWP

    stage_cost = (x, u) -> u[end]
    term_cost = (x, u) -> 100. * x[13 .+ (1:nwp)]' * x[13 .+ (1:nwp)]

    objective = [
        [Objective(stage_cost, nx, nu) for k = 1:N-1]...,
        Objective(term_cost, nx, 0)
    ]

    function fdyn(x, u)
        # λnext = progress_dynamics(x, u, nwp)
        λnext = u[4 + 13 .+ (1:nwp)]
        dt = u[end]
        xnext = u[4 .+ (1:13)]
        return [xnext; λnext; dt]
        # return [xnext; dt]
    end

    quad = Dynamics(fdyn, nx, nu)
    dynamics = [quad for k = 1:N-1]

    function constr_0(x, u)
        xnext = u[4 .+ (1:13)]
        λ = x[13 .+ (1:nwp)]
        μ = u[4 + 13 + nwp .+ (1:nwp)]
        sλ = u[4 + 13 .+ (1:nwp)]
        dt = u[end]
        return [
            xnext - x[1:13] + dt * quad_dynamics_continuous(x, u, m, J, Jinv, l, cτ);
            progress_constr(x, u, wps);
            λ - μ - sλ
        ]
    end
    function constr_k(x, u)
        xnext = u[4 .+ (1:13)]
        λ = x[13 .+ (1:nwp)]
        μ = u[4 + 13 + nwp .+ (1:nwp)]
        sλ = u[4 + 13 .+ (1:nwp)]
        dt = x[end]
        return [
            xnext - x[1:13] + dt * quad_dynamics_continuous(x, u, m, J, Jinv, l, cτ);
            progress_constr(x, u, wps);
            λ - μ - sλ
        ]
    end

    constraint_0 = Constraint(constr_0, nx, nu)
    constraint_k = Constraint(constr_k, nx, nu)
    constraints = [constraint_0; [constraint_k for k = 2:N-1]]
    
    #  
    ul = [Tmin * ones(4); -Inf * ones(2); zmin * ones(1); -Inf * ones(7); -ω_max_xy * ones(2); -ω_max_z * ones(1);
            zeros(nwp); zeros(nwp); zeros(3 * nwp - 1); 0.01 / N]
    uu = [Tmax * ones(4); Inf * ones(2); zmax * ones(1); Inf * ones(7); ω_max_xy * ones(2); ω_max_z * ones(1); 
            ones(nwp); ones(nwp); dtol^2 * ones(nwp); 0.1 * ones(nwp); ones(nwp - 1); 100. / N]
    bound = Bound(ul, uu)
    bounds = [bound for k in 1:N-1]

    # ## Initialise solver and solve

    solver = Solver(Float64, dynamics, objective, constraints, bounds, options=options)
    solver.options.verbose = verbose

    x1 = [p1; q1; zeros(3); zeros(3); ones(nwp); 0.0]

    # initialise variables
    
    wp_dists = cumsum([[norm(wps[1] - p1)]; [norm(wps[i] - wps[i-1]) for i = 2:nwp]])
    t_switch = Vector{Int64}(floor.(wp_dists * N / wp_dists[end]))
    t_guess = wp_dists[end] / vg / N

    i_wp = 1
    wp_next = wps[1]
    wp_last = p1
    x_inits = [[p1; q1; vg * (wp_next - wp_last) / norm(wp_next - wp_last); zeros(3)]]
    muinits = [zeros(nwp)]

    lamb_inits = [ones(nwp)]

    for t = 2:N-1
        if t > t_switch[i_wp]
            i_wp += 1
        end
        wp_last = i_wp == 1 ? p1 : wps[i_wp-1]
        wp_next = wps[i_wp]
        interp = 0.0
        if i_wp > 1
            interp = (t - t_switch[i_wp-1]) / (t_switch[i_wp] - t_switch[i_wp-1])
        else
            interp = t / t_switch[1]
        end
        pos_guess = (1-interp) * wp_last + interp * wp_next
        vel_guess = vg * (wp_next - wp_last) / norm(wp_next - wp_last)
        x_inits = [x_inits; [[pos_guess; q1; vel_guess; zeros(3)]]]

        if t + 1 == t_switch[i_wp]
            mu_guess = zeros(nwp)
            mu_guess[i_wp] = 0.0
            muinits = [muinits; [mu_guess]]
        else
            muinits = [muinits; [zeros(nwp)]]
        end

        lam_g = zeros(nwp)
        for i = 1:nwp
            if t+1 < t_switch[i]
              lam_g[i] = 1.0
            end
        end
        lamb_inits = [lamb_inits; [lam_g]]
    end

    ū = [[0.5 * Tmax .* ones(4); x_inits[k]; lamb_inits[k]; muinits[k]; 0.01 * ones(nwp); 0.001 * ones(nwp); ones(nwp - 1); t_guess] for k = 1:N-1]

    solver.options.max_iterations = 2000
    solve!(solver, x1, ū)

    x_sol, u_sol = get_trajectory(solver)
    # dt = u_sol[1][end]
    # for t = 1:N
    #     # println(x_sol[t][1:3], " ", x_sol[t][8:10], " ", u_sol[t][1:4])
    #     # println(t, " ", x_sol[t][1:13])
    #     # println(t, " ", x_sol[t][1:13] + dt * quad_dynamics_continuous(x_sol[t], u_sol[t], m, J, Jinv, l, cτ))
    #     # println(t, " ", x_inits[t][1:13])
    #     # println(t, " ", x_inits[t][1:13] + dt * quad_dynamics_continuous(x_sol[t], u_sol[t], m, J, Jinv, l, cτ))
    #     println(t, " ", x_sol[t][13 .+ (1:nwp)])
    #     # println(t, " ", u_sol[t][4:7])
    #     # println(t, " ", x_sol[t][end-3:end])
    # end
end