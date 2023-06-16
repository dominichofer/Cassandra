#include "Algorithm.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

Matrix CholeskyDecomposition(const Matrix& A)
{
    // Cholesky–Banachiewicz algorithm from
    // https://en.wikipedia.org/wiki/Cholesky_decomposition

    if (A.Rows() != A.Cols())
        throw std::runtime_error("Size mismatch");

    const std::size_t size = A.Rows();

    Matrix L(size, size);
    for (int i = 0; i < size; i++)
        for (int j = 0; j <= i; j++)
        {
            double sum = 0;
            for (std::size_t k = 0; k < j; k++)
                sum += L(i, k) * L(j, k);

            if (i == j)
                L(i, j) = std::sqrt(A(i, i) - sum);
            else
                L(i, j) = (A(i, j) - sum) / L(j, j);
        }
    return L;
}

Vector ForwardSubstitution(const Matrix& L, Vector b)
{
    for (int i = 0; i < b.size(); i++)
    {
        for (int j = 0; j < i; j++)
            b[i] -= L(i, j) * b[j];
        b[i] /= L(i, i);
    }
    return b;
}

Vector BackwardSubstitution(const Matrix& U, Vector b)
{
    for (int i = b.size() - 1; i >= 0; i--)
    {
        for (int j = i + 1; j < b.size(); j++)
            b[i] -= U(i, j) * b[j];
        b[i] /= U(i, i);
    }
    return b;
}

Vector GaussNewtonStep(
    const SymExp& function, const Vars& params, const Vars& vars,
    const std::ranges::random_access_range auto& x,
    const std::ranges::random_access_range auto& y,
    std::vector<double> param_values,
    double damping_factor)
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

    Matrix J(x_size, params.size()); // Jacobian matrix of residual function
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(J.Rows()); i++)
        for (std::size_t j = 0; j < J.Cols(); j++)
            J(i, j) = d_residual[j].At(vars, x[i]).value();
    Matrix J_T = transposed(J);

    auto residual_at_params = residual.At(params, param_values);
    std::vector<double> delta_y(y_size);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(y_size); i++)
        delta_y[i] = residual_at_params.At(vars, x[i]).At(Var{ "y" }, y[i]).value() * damping_factor;

    Matrix L = CholeskyDecomposition(J_T * J);
    return param_values - BackwardSubstitution(transposed(L), ForwardSubstitution(L, J_T * delta_y));
}

// Fits the 'params' of the non-linear 'function'
// so it fits the data 'x', 'y' in the sence of least squares,
// where 'x' corresponds to the 'variable' in 'function'.
Vector NonLinearLeastSquaresFit(
    const SymExp& function, const Vars& params, const Vars& vars,
    std::ranges::random_access_range auto&& x,
    std::ranges::random_access_range auto&& y,
    Vector param_values,
    int steps,
    double damping_factor)
{
    // From https://en.wikipedia.org/wiki/Non-linear_least_squares

    for (int i = 0; i < steps; i++)
    {
        std::vector old = param_values;
        param_values = GaussNewtonStep(function, params, vars, x, y, param_values);
        if (std::ranges::any_of(param_values, std::isnan<double>))
            return old;
        if (norm(param_values - old) < 1e-3 * norm(param_values))
            break;
    }
    return param_values;
}
