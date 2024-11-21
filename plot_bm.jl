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

plot(range(0, (N-1) * h, length=N), [x_ip v_ip x_al v_al], linecolor=[1 4 1 4], xtickfontsize=14, ytickfontsize=14, ylims=(0, 1.8),
    linestyle=[:solid :solid :dot :dot], linewidth=3, legendfontsize=12, legend=:best, background_color_legend = nothing,
    xlabel="\$t\$", xlabelfontsize=14,
    label=["\$y\$ (IPDDP2)" "\$v\$ (IPDDP2)" "\$y\$ (AL-iLQR)" "\$v\$ (AL-iLQR)"])
savefig("blockmove_yv.pdf")


ipddp_Fw = readdlm("ipddp_bm_Fw.txt", '\t', Float64, '\n')
alilqr_Fw = readdlm("alilqr_bm_Fw.txt", '\t', Float64, '\n')

F_ip = ipddp_Fw[:, 1]
w_ip = ipddp_Fw[:, 2]

F_al = alilqr_Fw[:, 1]
w_al = alilqr_Fw[:, 2]
s_al = alilqr_Fw[:, 3]

plot(range(0, (N-1) * h, length=N-1), [F_ip w_ip F_al w_al s_al], linecolor=[2 3 2 3 7], xtickfontsize=14, ytickfontsize=14,
    linestyle=[:solid :solid :dot :dot :dot], linewidth=3, legendfontsize=12, legend=:best, background_color_legend = nothing,
    xlabel="\$t\$", xlabelfontsize=14,
    label=["\$F\$          (IPDDP2)" L"$|Fv|$      (IPDDP2)" "\$F\$          (AL-iLQR)" "\$|Fw|\$     (AL-iLQR)" L"$s^{+} - s^{\_}$ (AL-iLQR)" ])
savefig("blockmove_Fw.pdf")
