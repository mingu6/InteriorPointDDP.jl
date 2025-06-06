function circleShape(xc, yc, r)
    θ = LinRange(0, 2*π, 500)
    xc .+ r * sin.(θ), yc .+ r * cos.(θ)
end

function plotCircle!(xc, yc, r)
    plot!(circleShape(xc, yc, r), seriestype = [:shape], lw = 0.5,
            c = :blue, linecolor = :black,
            legend = false, fillalpha = 0.2, aspect_ratio = 1)
end

function plotTrajectory!(x̄)
    x = map(x -> x[1], x̄)
    y = map(x -> x[2], x̄)
    plot!(x, y, linealpha=0.8, linecolor=1, linewidth=2)
end
