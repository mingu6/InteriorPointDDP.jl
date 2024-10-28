function solver_info()
    println(crayon"bold red","   
    ██╗██████╗░█████╗░░░█████╗░░░██████╗░
    ██║██╔══██╗██╔══██╗░██╔══██╗░██╔══██╗
    ██║██████╔╝██║░░░██║██║░░░██║██████╔╝
    ██║██╔═══╝░██║░░██║░██║░░██║░██╔═══╝░
    ██║██║░░░░░█████╔╝░░█████╔╝░░██║░░░░░
    ╚═╝╚═╝░░░░░╚════╝░░░╚════╝░░░╚═╝░░░░░    
    ")
    println(crayon"reset bold black",
    "Ming Xu, Stephen Gould, Iman Shames")
    println("Implemented by Mingda Xu, Pranav Pativada and Jeffrey Liang")
    println("School of Computing and School of Engineering")
    println("The Australian National University\n")
    print(crayon"reset")
end

function iteration_status(
    data::SolverData,
    options::Options
    )

    # header
    if rem(data.k, options.print_frequency) == 0
        @printf "  iter     objective        pr_inf       du_inf       cs_inf     lg(μ)   lg(reg)    alpha     soc     ls   wall_time \n"
    end
    

    # iteration information
    @printf(" %5s   %.8e   %.4e   %.4e   %.4e   % 1.2f  %5s   %.4e  %2s   %2s   %5.2f \n",
        data.k, data.objective, data.primal_inf, data.dual_inf, data.cs_inf,
        log10(data.μ), data.reg_last == 0.0 ? "    -  " : @sprintf("% 2.4f", log10(data.reg_last)),
        data.step_size, data.p, data.l, data.wall_time * 1000)
end

function on_exit(data::SolverData)
    println()
    println()
    if data.status == 0
        println("EXIT: Optimal solution found.")
    elseif data.status == 1
        println("EXIT: Failed, unable to find iteration matrix with desired inertia in backward pass.")
    elseif data.status == 7
        println("EXIT: Failed, line-search unable to find acceptable iterate in forward pass.")
    elseif data.status == 8
        println("EXIT: Failed, maximum solver iterations reached.")
    else
        println("DEBUG: This message should not display.")
    end
    println()
end
