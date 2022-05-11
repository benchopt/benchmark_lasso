using Lasso
using PyCall
using SparseArrays

function scipyCSC_to_julia(A)
    m, n = A.shape
    colPtr = Int[i + 1 for i in PyArray(A."indptr")]
    rowVal = Int[i + 1 for i in PyArray(A."indices")]
    nzVal = Vector{Float64}(PyArray(A."data"))
    B = SparseMatrixCSC{Float64,Int}(m, n, colPtr, rowVal, nzVal)

    return B
end

function solve_lasso(
    X, y::Vector{Float64}, lambda::Vector{Float64}, fit_intercept::Bool, tol::Float64
)
    lasso_fit = fit(
        LassoPath,
        X,
        y;
        λ=lambda,
        standardize=false,
        intercept=fit_intercept,
        randomize=false,
        stopearly=false,
        maxncoef=max(size(X, 1), size(X, 2)) * 100,
        cd_tol=tol,
    )
    w = coef(lasso_fit)

    return w
end
