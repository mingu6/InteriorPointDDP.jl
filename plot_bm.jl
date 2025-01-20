using DelimitedFiles
using Plots
using LaTeXStrings

ipddp_xv = readdlm("ipddp_bm_xv.txt", '\t', Float64, '\n')
alilqr_xv = readdlm("alilqr_bm_xv.txt", '\t', Float64, '\n')

x_ip = ipddp_xv[:, 1]
v_ip = ipddp_xv[:, 2]

x_al = alilqr_xv[:, 1]
v_al = alilqr_xv[:, 2]

N = length(x_ip)
h = 0.01

plot(range(0, (N-1) * h, length=N), [x_ip x_al], linecolor=[1 1], xtickfontsize=14, ytickfontsize=14, #ylims=(0, 1.8),
    linestyle=[:solid :dot], linewidth=3, legendfontsize=16, legend=:best, background_color_legend = nothing,
    xlabel="\$t\$", xlabelfontsize=16,
    label=["\$y_t\$ (IPDDP2)" "\$y_t\$ (AL-iLQR)"])
savefig("blockmove_y.pdf")

plot(range(0, (N-1) * h, length=N), [v_ip v_al], linecolor=[2 2], xtickfontsize=14, ytickfontsize=14, #ylims=(0, 1.9),
    linestyle=[:solid :dot], linewidth=3, legendfontsize=16, legend=:best, background_color_legend = nothing,
    xlabel="\$t\$", xlabelfontsize=16,
    label=["\$v_t\$ (IPDDP2)" "\$v_t\$ (AL-iLQR)"])
savefig("blockmove_v.pdf")


ipddp_Fw = readdlm("ipddp_bm_Fw.txt", '\t', Float64, '\n')
alilqr_Fw = readdlm("alilqr_bm_Fw.txt", '\t', Float64, '\n')

F_ip = ipddp_Fw[:, 1]
w_ip = ipddp_Fw[:, 2]

F_al = alilqr_Fw[:, 1]
w_al = alilqr_Fw[:, 2]
s_al = alilqr_Fw[:, 3]

plot(range(0, (N-1) * h, length=N-1), [F_ip F_al], linecolor=[3 3], xtickfontsize=14, ytickfontsize=14,
    linestyle=[:solid :dot], linewidth=3, legendfontsize=16, legend=:best, background_color_legend = nothing,
    xlabel="\$t\$", xlabelfontsize=16,
    label=["\$F_t\$ (IPDDP2)" "\$F_t\$ (AL-iLQR)"])
savefig("blockmove_F.pdf")

plot(range(0, (N-1) * h, length=N-1), [w_ip w_al], linecolor=[4 4], xtickfontsize=14, ytickfontsize=14, #ylims=(-3, 18),
    linestyle=[:solid :dot], linewidth=3, legendfontsize=16, legend=:best, background_color_legend = nothing,
    xlabel="\$t\$", xlabelfontsize=16,
    label=[ "\$|F_tv_t|\$        (IPDDP2)" "\$|F_tv_t|\$       (AL-iLQR)"])
savefig("blockmove_Fv.pdf")
