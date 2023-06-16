#pragma once
#include "Matrix.h"
#include "SymExp.h"
#include "Vector.h"
#include <ranges>

// Lower triangular matrix of the Cholesky decomposition.
// A: hermitian positive-definite matrix
Matrix CholeskyDecomposition(const Matrix& A);

Vector ForwardSubstitution(const Matrix&, Vector);
Vector BackwardSubstitution(const Matrix&, Vector);

// Performs a step of the Gauss–Newton algorithm
Vector GaussNewtonStep(
    const SymExp& function, const Vars& params, const Vars& vars,
    const std::ranges::random_access_range auto& x,
    const std::ranges::random_access_range auto& y,
    std::vector<double> param_values,
    double damping_factor = 1.0);

// Fits the 'params' of the non-linear 'function'
// so it fits the data 'x', 'y' in the sence of least squares,
// where 'x' corresponds to the 'variable' in 'function'.
Vector NonLinearLeastSquaresFit(
    const SymExp& function, const Vars& params, const Vars& vars,
    std::ranges::random_access_range auto&& x,
    std::ranges::random_access_range auto&& y,
    Vector param_values,
    int steps,
    double damping_factor = 1.0);
