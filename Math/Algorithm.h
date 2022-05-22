#pragma once
#include "Vector.h"
#include "DenseMatrix.h"
#include "SymExp.h"
#include <cmath>
#include <stdexcept>
#include <ranges>

// Lower triangular matrix of the Cholesky decomposition.
// A: hermitian positive-definite matrix
template <typename T>
DenseMatrix<T> CholeskyDecomposition(const DenseMatrix<T>& A)
{
    // Cholesky–Banachiewicz algorithm from
    // https://en.wikipedia.org/wiki/Cholesky_decomposition

    if (A.Rows() != A.Cols())
        throw std::runtime_error("Size mismatch");

    const std::size_t size = A.Rows();

    DenseMatrix<T> L(size, size);
    for (int i = 0; i < size; i++)
        for (int j = 0; j <= i; j++)
        {
            T sum = 0;
            for (std::size_t k = 0; k < j; k++)
                sum += L(i, k) * L(j, k);

            if (i == j)
                L(i, j) = std::sqrt(A(i, j) - sum);
            else
                L(i, j) = (A(i, j) - sum) / L(j, j);
        }
    return L;
}

template <typename M, typename V>
std::valarray<V> ForwardSubstitution(const DenseMatrix<M>& L, std::valarray<V> b)
{
    for (int i = 0; i < b.size(); i++)
    {
        for (int j = 0; j < i; j++)
            b[i] -= L(i, j) * b[j];
        b[i] /= L(i, i);
    }
    return b;
}

template <typename M, typename V>
std::valarray<V> BackwardSubstitution(const DenseMatrix<M>& U, std::valarray<V> b)
{
    for (int i = b.size() - 1; i >= 0; i--)
    {
        for (int j = i + 1; j < b.size(); j++)
            b[i] -= U(i, j) * b[j];
        b[i] /= U(i, i);
    }
    return b;
}

// Performs a step of the Gauss–Newton algorithm
std::valarray<double> GaussNewtonStep(
    const SymExp& function, const Vars& params, const Vars& vars,
    const std::ranges::random_access_range auto& x,
    const std::ranges::random_access_range auto& y,
    std::valarray<double> param_values,
    double damping_factor = 1.0)
{
    // From https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm

    if (params.size() != param_values.size())
        throw std::runtime_error("Size mismatch!");
    std::size_t x_size = std::ranges::size(x);
    std::size_t y_size = std::ranges::size(y);
    if (x_size != y_size)
        throw std::runtime_error("Size mismatch!");

    SymExp residual = Var{ "y" } - function;
    SymExps d_residual = residual.DeriveAt(params, param_values);

    DenseMatrix<double> J(x_size, params.size()); // Jacobian matrix of residual function
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(J.Rows()); i++)
        for (std::size_t j = 0; j < J.Cols(); j++)
            J(i, j) = d_residual[j].At(vars, x[i]).value();
    DenseMatrix J_T = transposed(J);

    auto residual_at_params = residual.At(params, param_values);
    std::valarray<double> delta_y(y_size);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < y_size; i++)
        delta_y[i] = residual_at_params.At(vars, x[i]).At(Var{ "y" }, y[i]).value() * damping_factor;

    DenseMatrix L = CholeskyDecomposition(J_T * J);
    return param_values - BackwardSubstitution(transposed(L), ForwardSubstitution(L, J_T * delta_y));
}

// Fits the 'params' of the non-linear 'function'
// so it fits the data 'x', 'y' in the sence of least squares,
// where 'x' corresponds to the 'variable' in 'function'.
std::valarray<double> NonLinearLeastSquaresFit(
    const SymExp& function, const Vars& params, const Vars& vars,
    std::ranges::random_access_range auto&& x,
    std::ranges::random_access_range auto&& y,
    std::valarray<double> param_values,
    int steps = 1'000,
    double damping_factor = 1.0)
{
    // From https://en.wikipedia.org/wiki/Non-linear_least_squares

    for (int i = 0; i < steps; i++)
    {
        std::valarray old = param_values;
        param_values = GaussNewtonStep(function, params, vars, x, y, param_values);
        if (std::ranges::any_of(param_values, std::isnan<double>))
            return old;
        if (norm(param_values - old) < 1e-3 * norm(param_values))
            break;
    }
    return param_values;
}