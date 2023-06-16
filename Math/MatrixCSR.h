#pragma once
#include "Vector.h"
#include "Matrix.h"
#include <cstdint>
#include <span>
#include <vector>

// Compressed Sparse Row Matrix
// With fixed number of non-zero elements per row. With only 1 as element but the matrix can have multiple entries per element.
class MatrixCSR
{
	const std::size_t elements_per_row; // number of non-zero elements in a row
	const std::size_t cols;
	std::vector<uint32_t> col_indices;

	friend Vector JacobiPreconditionerOfATA(const MatrixCSR&);
public:
	MatrixCSR(std::size_t elements_per_row, std::size_t rows, std::size_t cols);

	std::size_t Rows() const noexcept { return col_indices.size() / elements_per_row; }
	std::size_t Cols() const noexcept { return cols; }

	std::span<      uint32_t> Row(std::size_t index)       noexcept;
	std::span<const uint32_t> Row(std::size_t index) const noexcept;
};

// Jacobi Preconditioner of A' * A
// Returns 1 / diag(A' * A)
Vector JacobiPreconditionerOfATA(const MatrixCSR&);

Vector operator*(const MatrixCSR&, const Vector&);
Vector operator*(const Vector&, const MatrixCSR&);
