#include "Algorithm.h"
#include <algorithm>
#include <cassert>
#include <cmath>

Vector sqrt(Vector x)
{
    const int64_t size = static_cast<int64_t>(x.size());
    #pragma omp parallel for
    for (int64_t i = 0; i < size; i++)
        x[i] = sqrt(x[i]);
    return x;
}

Vector::value_type dot(const Vector& a, const Vector& b)
{
    assert(a.size() == b.size());

    const int64_t size = static_cast<int64_t>(a.size());
    Vector::value_type sum = 0;
    #pragma omp parallel for reduction(+ : sum)
    for (int64_t i = 0; i < size; i++)
        sum += a[i] * b[i];
    return sum;
}

Vector::value_type norm(const Vector& v)
{
    return std::sqrt(dot(v, v));
}


std::tuple<double, Vector> decompose(const Vector& x)
{
    const auto n = norm(x);
    return std::tuple(n, x / n);
}

DenseMatrix CholeskyDecomposition(const DenseMatrix& A)
{
    // Cholesky–Banachiewicz algorithm from
    // https://en.wikipedia.org/wiki/Cholesky_decomposition

    assert(A.Rows() == A.Cols());
    const std::size_t size = A.Rows();

    DenseMatrix L(size, size);
    for (int i = 0; i < size; i++)
        for (int j = 0; j <= i; j++)
        {
            DenseMatrix::value_type sum = 0;
            for (int k = 0; k < j; k++)
                sum += L(i, k) * L(j, k);

            if (i == j)
                L(i, j) = std::sqrt(A(i, j) - sum);
            else
                L(i, j) = (A(i, j) - sum) / L(j, j);
        }
    return L;
}

Vector ForwardSubstitution(const DenseMatrix& L, Vector b)
{
    for (int i = 0; i < b.size(); i++)
    {
        for (int j = 0; j < i; j++)
            b[i] -= L(i, j) * b[j];
        b[i] /= L(i, i);
    }
    return b;
}

Vector BackwardSubstitution(const DenseMatrix& U, Vector b)
{
    for (int i = b.size() - 1; i >= 0; i--)
    {
        for (int j = i + 1; j < b.size(); j++)
            b[i] -= U(i, j) * b[j];
        b[i] /= U(i, i);
    }
    return b;
}

Vector GaussNewtonStep(const SymExp& function, const Vars& params, const Vars& vars,
    const Vector& param_values,
    const std::vector<Vector>& x, const Vector& y)
{
    // From https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm

    if (params.size() != param_values.size())
        throw std::runtime_error("Size mismatch!");
    if (x.size() != y.size())
        throw std::runtime_error("Size mismatch!");

    SymExp residual = Var{ "y" } - function;
    SymExps dr = residual.DeriveAt(params, param_values);

    DenseMatrix J(x.size(), params.size()); // Jacobian matrix of residual function
    for (std::size_t i = 0; i < J.Rows(); i++)
        for (std::size_t j = 0; j < J.Cols(); j++)
            J(i, j) = dr[j].At(vars, x[i]).value();
    DenseMatrix J_T = transposed(J);

    Vector delta_y(y.size());
    auto residual_at_params = residual.At(params, param_values);
    for (std::size_t i = 0; i < y.size(); i++)
        delta_y[i] = residual_at_params.At(vars, x[i]).At(Var{ "y" }, y[i]).value();

    DenseMatrix L = CholeskyDecomposition(J_T * J);
    return param_values - BackwardSubstitution(transposed(L), ForwardSubstitution(L, J_T * delta_y));
}

// Fits the 'params' of the non-linear 'function'
// so it fits the data 'x', 'y' in the sence of least squares,
// where 'x' corresponds to the 'variable' in 'function'.
Vector NonLinearLeastSquaresFit(const SymExp& function, const Vars& params, const Vars& vars,
    const std::vector<Vector>& x, const Vector& y,
    Vector params_values)
{
    // From https://en.wikipedia.org/wiki/Non-linear_least_squares
    for (int i = 0; i < 100; i++)
    {
        Vector old = params_values;
        params_values = GaussNewtonStep(function, params, vars, params_values, x, y);

        if (std::any_of(params_values.begin(), params_values.end(), [](auto x) { return std::isnan(x); }))
            return old;
        if (norm(params_values - old) < 1e-3 * norm(params_values))
            break;
    }
    return params_values;
}