using Core
using LinearAlgebra

st(w::Vector{Float64}, t::Float64) = @. sign(w) * max(abs(w) - t, 0.)

function solve_lasso(
    X::Matrix{Float64}, 
    y::Vector{Float64}, 
    lambda::Float64,
    n_iter::Int
)
    L = opnorm(X)^2
    p = size(X, 2)
    w = zeros(p)
    z = copy(w)
    zold = copy(z)
    residual = similar(y)
    diffvec = similar(w)
    gradient = similar(w)  
    t = 1.
    told = 1.              
    for i ∈ 1:n_iter
        zold = copy(z)
        told = t
        residual = X * w - y
        gradient = X' * residual
        diffvec = w - gradient / L
        z = st(diffvec, lambda / L)
        t = (1. + sqrt(1. + 4. * told^2)) / 2.
        w = z + ((told - 1.) / t) * (z - zold)
    end

    return w
end