# mutable struct InteriorPoint{T,C,CX,CU}
#     objective::Objective{T}
#     constraint_data::ConstraintsData{T,C,CX,CU}
# end

# function interior_point(model::Model{T}, objective::Objective{T}, constraints::Constraints{T}) where T
#     # horizon
#     H = length(model) + 1
#     data = constraint_data(model, constraints)
#     InteriorPoint(objective, 
#         data)
# end

# function cost(interior_point::InteriorPoint, states, actions, parameters)
#     # objective
#     J = cost(interior_point.objective, states, actions, parameters)
#     return J
# end

# Base.length(objective::InteriorPoint) = length(objective.objective)