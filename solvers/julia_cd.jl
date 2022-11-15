using Core
using LinearAlgebra

st(w::Float64, t::Float64) = sign(w) * max(abs(w) - t, 0.)

function solve_lasso(
    X::Matrix{Float64}, 
    y::Vector{Float64}, 
    lmbd::Float64,
    L::Vector{Float64},
    n_iter::Int,
)
    p = size(X, 2)
    r = copy(y)
    w = zeros(p)
    for i in 1:n_iter
        for (j, Xj) in enumerate(eachcol(X))
            (L[j] == 0.) && continue
            wj = w[j]
            w[j] = st(wj + (Xj' * r) / L[j], lmbd / L[j])
            diff = wj - w[j]
            if diff != 0.
                r += diff * Xj
            end
        end
    end

    return w
end
