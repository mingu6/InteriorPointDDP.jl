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
    "Andrei Pavlov, Iman Shames, and Chris Manzie")
    println("Implemented by Mingda Xu, Pranav Pativada and Jeffrey Liang")
    println("School of Computing")
    println("The Australian National University\n")
    print(crayon"reset")
end

function iteration_status(
    data::SolverData,
    options::Options
    )

    # header
    if rem(data.k, options.print_frequency) == 0
        @printf "  iter   objective      pr_inf       du_inf       cs_inf     lg(μ)   lg(ϕ)    alpha     ls   wall_time \n"
    end
    

    # iteration information
    @printf(" %5s   %.4e   %.4e   %.4e   %.4e   % 1.2f  %5s   %.4e  %2s   %5.2f \n",
        data.k, data.objective, data.primal_inf, data.dual_inf, data.cs_inf,
        log10(data.μ), data.reg_last == 0.0 ? "-" : @sprintf("% 2.2f", log10(data.reg_last)),
        data.step_size, data.l, data.wall_time * 1000)
end
