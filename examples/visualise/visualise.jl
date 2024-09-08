using GeometryBasics
using Rotations
using CoordinateTransformations
using LinearAlgebra

function cable_transform(y, z)
    v1 = [0.0, 0.0, 1.0]
    v2 = y[1:3,1] - z[1:3,1]
    normalize!(v2)
    ax = cross(v1, v2)
    ang = acos(v1'*v2)
    R = AngleAxis(ang, ax...)

    if any(isnan.(R))
        R = I
    else
        nothing
    end

    compose(Translation(z), LinearMap(R))
end

function default_background!(vis)
    setvisible!(vis["/Background"], true)
    setprop!(vis["/Background"], "top_color", RGBA(1.0, 1.0, 1.0, 1.0))
    setprop!(vis["/Background"], "bottom_color", RGBA(1.0, 1.0, 1.0, 1.0))
    setvisible!(vis["/Axes"], false)
end

function state_to_configuration(x::Vector{Vector{T}}) where T 
	N = length(x) 
	nq = convert(Int, floor(length(x[1]) / 2))
	q = Vector{T}[] 

	for k = 1:N 
		if k == 1 
			push!(q, x[k][1:nq]) 
		end
		push!(q, x[k][nq .+ (1:nq)])
	end
	
	return q 
end
