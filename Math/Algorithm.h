#pragma once
#include <cassert>
#include <cmath>
#include <tuple>
#include "Vector.h"
#include "DenseMatrix.h"
#include "SymExp.h"

inline Vector sqrt(Vector x)
{
	const int64_t size = static_cast<int64_t>(x.size());
	#pragma omp parallel for
	for (int64_t i = 0; i < size; i++)
		x[i] = sqrt(x[i]);
	return x;
}

inline Vector::value_type dot(const Vector& a, const Vector& b)
{
	assert(a.size() == b.size());

	const int64_t size = static_cast<int64_t>(a.size());
	Vector::value_type sum = 0;
	#pragma omp parallel for reduction(+ : sum)
	for (int64_t i = 0; i < size; i++)
		sum += a[i] * b[i];
	return sum;
}

inline Vector::value_type norm(const Vector& v)
{
	return std::sqrt(dot(v, v));
}


inline std::tuple<double, Vector> decompose(const Vector& x)
{
	const auto n = norm(x);
	return std::tuple(n, x / n);
}

// Lower triangular matrix of the Cholesky decomposition.
// A: hermitian positive-definite matrix
template <typename T>
DenseMatrix<T> CholeskyDecomposition(const DenseMatrix<T>& A)
{
    // Cholesky–Banachiewicz algorithm from
    // https://en.wikipedia.org/wiki/Cholesky_decomposition

    assert(A.Rows() == A.Cols());
    const std::size_t size = A.Rows();

    DenseMatrix<T> L(size, size);
    for (int i = 0; i < size; i++)
        for (int j = 0; j <= i; j++)
        {
            T sum = 0;
            for (int k = 0; k < j; k++)
                sum += L(i, k) * L(j, k);

            if (i == j)
                L(i, j) = std::sqrt(A(i, j) - sum);
            else
                L(i, j) = 1.0 / L(j, j) * (A(i, j) - sum);
        }
    return L;
}

template <typename T>
Vector ForwardSubstitution(const DenseMatrix<T>& L, Vector b)
{
    for (int i = 0; i < b.size(); i++)
    {
        for (int j = 0; j < i; j++)
            b[i] -= L(i, j) * b[j];
        b[i] /= L(i, i);
    }
    return b;
}

template <typename T>
Vector BackwardSubstitution(const DenseMatrix<T>& U, Vector b)
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
inline Vector GaussNewtonStep(const SymExp& function, const Vars& params, const Vars& vars,
                       const Vector& param_values,
                       const std::vector<Vector>& x, const Vector& y)
{
    if (params.size() != param_values.size())
        throw std::runtime_error("Size mismatch!");
    if (x.size() != y.size())
        throw std::runtime_error("Size mismatch!");

    SymExp residual = Var{"y"} - function;
    SymExps dr = residual.DeriveAt(params, param_values);

    DenseMatrix<float> J(x.size(), params.size()); // Jacobian matrix of residual function
    for (std::size_t i = 0; i < J.Rows(); i++)
        for (std::size_t j = 0; j < J.Cols(); j++)
            J(i, j) = dr[j].At(vars, x[i]).value();
    DenseMatrix<float> J_T = transposed(J);

    auto residual_at_params = residual.At(params, param_values);

    Vector delta_y(y.size());
    for (std::size_t i = 0; i < y.size(); i++)
        delta_y[i] = residual_at_params.At(vars, x[i]).At(Var{"y"}, y[i]).value();

    DenseMatrix<float> L = CholeskyDecomposition(J_T * J);
    return param_values - BackwardSubstitution(transposed(L), ForwardSubstitution(L, J_T * delta_y));
}
#include <iostream>

// Fits the 'params' of the non-linear 'function'
// so it fits the data 'x', 'y' in the sence of least squares,
// where 'x' corresponds to the 'variable' in 'function'.
inline Vector NonLinearLeastSquaresFit(const SymExp& function, const Vars& params, const Vars& vars,
                                const std::vector<Vector>& x, const std::vector<float>& y,
                                Vector params_values)
{
    // From https://en.wikipedia.org/wiki/Non-linear_least_squares
    for (int i = 0; i < 100; i++)
    {
        Vector old = params_values;
        for (auto p : params_values)
            std::cout << p << ", "; 
        std::cout << std::endl;
        params_values = GaussNewtonStep(function, params, vars, params_values, x, y);
        if (norm(params_values - old) < 1e-3 * norm(params_values))
            break;
    }
    return params_values;
}