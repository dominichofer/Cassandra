#pragma once

template <typename T>
DenseMatrix<T> Matrix(T a, T b, T c, T d)
{
	DenseMatrix<T> mat(2, 2);
	mat(0, 0) = a;
	mat(0, 1) = b;
	mat(1, 0) = c;
	mat(1, 1) = d;
	return mat;
}