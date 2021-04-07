#pragma once
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <numeric>
#include <vector>
#include <limits>
#include <stdexcept>
#include "Vector.h"
#include "MatrixInterface.h"

// Compressed Sparse Row Matrix
// With fixed number of elements per row. With only 1 as element but the matrix can have multiple entries per element.
template <typename IndexType>
class MatrixCSR : public Matrix
{
public:
	using index_type = IndexType;
private:
	const std::size_t row_size; // number of elements in a row
	const std::size_t cols;
	std::vector<index_type> col_indices;
public:
	MatrixCSR(std::size_t row_size, std::size_t cols, std::size_t rows)
		: row_size(row_size)
		, cols(cols)
		, col_indices(rows * row_size)
	{
		if (std::numeric_limits<index_type>::max() < cols)
			throw std::runtime_error("Template type 'index_type' is too small to represent 'cols'.");
	}

	std::size_t Rows() const noexcept override { return col_indices.size() / row_size; }
	std::size_t Cols() const noexcept override { return cols; }

	auto begin() noexcept { return col_indices.begin(); }
	auto begin() const noexcept { return col_indices.begin(); }
	auto cbegin() const noexcept { return col_indices.cbegin(); }
	auto end() noexcept { return col_indices.end(); }
	auto end() const noexcept { return col_indices.end(); }
	auto cend() const noexcept { return col_indices.cend(); }

	auto begin(std::size_t row) noexcept { return col_indices.begin() + row * row_size; }
	auto begin(std::size_t row) const noexcept { return col_indices.begin() + row * row_size; }
	auto cbegin(std::size_t row) const noexcept { return col_indices.cbegin() + row * row_size; }
	auto end(std::size_t row) noexcept { return col_indices.begin() + (row + 1) * row_size; }
	auto end(std::size_t row) const noexcept { return col_indices.begin() + (row + 1) * row_size; }
	auto cend(std::size_t row) const noexcept { return col_indices.cbegin() + (row + 1) * row_size; }

	Vector Ax(const Vector& x) const override
	{
		if (x.size() != Cols())
			throw std::runtime_error("Size mismatch.");
		//if (row_size % 4 != 0)
		//	throw std::runtime_error("'row_size' is not equivalent to 0 (mod 4).");

		const int64_t rows = static_cast<int64_t>(Rows());
		Vector result(rows, 0);
		#pragma omp parallel for schedule(dynamic,64)
		for (int64_t i = 0; i < rows; i++)
		{
			double sum = 0; // prevents cache thrashing
			//for (auto j = i * row_size; j < (i + 1) * row_size; j+=4)
			//	sum += x[col_indices[j+0]] + x[col_indices[j+1]] + x[col_indices[j+2]] + x[col_indices[j+3]]; // TODO: Test if unrolling is worth it!
			for (auto j = i * row_size; j < (i + 1) * row_size; j++)
				sum += x[col_indices[j]];
			result[i] = static_cast<Vector::value_type>(sum);
		}
		return result;
	}

	Vector ATx(const Vector& x) const override
	{
		if (x.size() != Rows())
			throw std::runtime_error("Size mismatch.");

		const int64_t rows = static_cast<int64_t>(Rows());
		const int64_t cols = static_cast<int64_t>(Cols());
		Vector result(cols, 0);
		#pragma omp parallel
		{
			Vector local_result(cols, 0); // prevents cache thrashing
			#pragma omp for nowait schedule(dynamic,64)
			for (int64_t i = 0; i < rows; i++)
				for (auto j = i * row_size; j < (i + 1) * row_size; j++)
					local_result[col_indices[j]] += x[i];

			#pragma omp critical
			{
				result += local_result;
			}
		}
		return result;
	}
	
	// Returns diag(A' * A)
	Vector DiagATA() const
	{
		Vector ret(cols, 0);
		#pragma omp parallel
		{
			Vector local_ret(cols, 0); // prevents cache thrashing

			#pragma omp for nowait schedule(dynamic,64)
			for (int64_t i = 0; i < col_indices.size(); i++)
				local_ret[col_indices[i]] += 1.0;

			#pragma omp critical
			{
				ret += local_ret;
			}
		}
		return ret;
	}

	// Jacobi Preconditioner Square
	// Returns 1 / diag(A' * A)
	Vector JacobiPreconditionerSquare(Vector::value_type infinity = std::numeric_limits<Vector::value_type>::infinity()) const
	{
		return inv(DiagATA(), infinity);
	}
};