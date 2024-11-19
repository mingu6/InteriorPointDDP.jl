function parse_results_ipopt(output::String)
    regex_obj = r"Objective...............:   (\d+.\d+e[+]\d+)    (\d+.\d+e[+]\d+)"
    regex_constr = r"Constraint violation....:   (\d+.\d+e[+]\d+)    (\d+.\d+e[+]\d+)"
    regex_niter = r"Number of Iterations....: (\d+)"
    regex_succ = r"EXIT: Optimal Solution Found."
    regex_solver = r"Total seconds in IPOPT[\s\S]*=\s+(\d+.\d+)"
    regex_fn_eval = r"Total seconds in NLP function evaluations            =\s+(\d+.\d+)"
    objective = Float64(0.0)
    constr_viol = Float64(0.0)
    n_iter = Int64(0)
    solver_time = Float64(0.0)
    fn_eval_time = Float64(0.0)
    succ = false
    for line in Base.eachsplit(output, "\n")
        obj_match = match(regex_obj, line)
        constr_match = match(regex_constr, line)
        niter_match = match(regex_niter, line)
        succ_match = match(regex_succ, line)
        solver_match = match(regex_solver, line)
        fn_eval_match = match(regex_fn_eval, line)
        if !isnothing(obj_match)
            objective = parse(Float64, obj_match.captures[2])
        end
        if !isnothing(constr_match)
            constr_viol = parse(Float64, constr_match.captures[2])
        end
        if !isnothing(niter_match)
            n_iter = parse(Int64, niter_match.captures[1])
        end
        if !isnothing(succ_match)
            succ = true
        end
        if !isnothing(solver_match)
            solver_time = parse(Float64, solver_match.captures[1])
        end
        if !isnothing(fn_eval_match)
            fn_eval_time = parse(Float64, fn_eval_match.captures[1])
        end
    end
    return objective, constr_viol, n_iter, succ, solver_time * 1000, (solver_time + fn_eval_time) * 1000
end
