struct Cartpole{T}
    # dimensions
    nq::Int # generalized coordinates
    nu::Int # controls

    mc::T     # mass of the cart in kg
    mp::T     # mass of the pole (point mass at the end) in kg
    l::T      # length of the pole in m
    g::T      # gravity m/s^2
end

cartpole = Cartpole(2, 1, 1.0, 0.3, 0.5, 9.81)