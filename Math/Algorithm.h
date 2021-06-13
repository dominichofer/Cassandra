#pragma once
#include "Vector.h"
#include "DenseMatrix.h"
#include "SymExp.h"
#include <tuple>

Vector sqrt(Vector);
Vector::value_type dot(const Vector&, const Vector&);
Vector::value_type norm(const Vector&);
std::tuple<double, Vector> decompose(const Vector&);

// Lower triangular matrix of the Cholesky decomposition.
// A: hermitian positive-definite matrix
DenseMatrix CholeskyDecomposition(const DenseMatrix& A);

Vector ForwardSubstitution(const DenseMatrix& L, Vector);
Vector BackwardSubstitution(const DenseMatrix& U, Vector);

// Performs a step of the Gauss–Newton algorithm
Vector GaussNewtonStep(const SymExp& function, const Vars& params, const Vars& vars,
    const Vector& param_values,
    const std::vector<Vector>& x, const Vector& y);

// Fits the 'params' of the non-linear 'function'
// so it fits the data 'x', 'y' in the sence of least squares,
// where 'x' corresponds to the 'variable' in 'function'.
Vector NonLinearLeastSquaresFit(const SymExp& function, const Vars& params, const Vars& vars,
    const std::vector<Vector>& x, const Vector& y,
    Vector params_values);