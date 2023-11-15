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
    total_iterations,
    outer_iterations,
    inner_iterations,
    residual_violation,
    constraint_violation,
    penalty,
    step_size,
    print_frequency,
    )

    # header
    if rem(total_iterations - 1, print_frequency) == 0
        @printf "------------------------------------------------------------------------------------------------\n"
        @printf "total  outer  inner |residual| |constraint|   penalty   step  \n"
        @printf "------------------------------------------------------------------------------------------------\n"
    end

    # iteration information
    @printf("%3d     %2d    %3d   %9.2e  %9.2e   %9.2e   %9.2e \n",
        total_iterations,
        outer_iterations,
        inner_iterations,
        residual_violation,
        equality_violation,
        penalty,
        step_size)
end
