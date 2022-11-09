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
    residual = similar(y)
    diffvec = similar(w)
    gradient = similar(w)                  
    for i ∈ 1:n_iter
        residual = X * w - y
        gradient = X' * residual
        diffvec = w - gradient / L
        w = st(diffvec, lambda / L)
    end

    return w
end
