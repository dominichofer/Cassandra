#pragma once
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <numeric>
#include <vector>
#include <limits>
#include <stdexcept>
#include <span>
#include "Vector.h"
#include "Matrix.h"

// Compressed Sparse Row Matrix
// With fixed number of non-zero elements per row. With only 1 as element but the matrix can have multiple entries per element.
template <typename IndexType>
class MatrixCSR
{
public:
	using index_type = IndexType;
private:
	const std::size_t elements_per_row; // number of non-zero elements in a row
	const std::size_t cols;
	std::vector<index_type> col_indices;
public:
	MatrixCSR(std::size_t elements_per_row, std::size_t cols, std::size_t rows) // TODO: Swap cols and rows!
		: elements_per_row(elements_per_row)
		, cols(cols)
		, col_indices(rows * elements_per_row)
	{
		if (std::numeric_limits<index_type>::max() < cols)
			throw std::runtime_error("Template type 'IndexType' is too small to represent 'cols'.");
	}

	std::size_t Rows() const noexcept { return col_indices.size() / elements_per_row; }
	std::size_t Cols() const noexcept { return cols; }

	auto begin() noexcept { return col_indices.begin(); }
	auto begin() const noexcept { return col_indices.begin(); }
	auto cbegin() const noexcept { return col_indices.cbegin(); }
	auto end() noexcept { return col_indices.end(); }
	auto end() const noexcept { return col_indices.end(); }
	auto cend() const noexcept { return col_indices.cend(); }

	auto begin(std::size_t row) noexcept { return col_indices.begin() + row * elements_per_row; }
	auto begin(std::size_t row) const noexcept { return col_indices.begin() + row * elements_per_row; }
	auto cbegin(std::size_t row) const noexcept { return col_indices.cbegin() + row * elements_per_row; }
	auto end(std::size_t row) noexcept { return col_indices.begin() + (row + 1) * elements_per_row; }
	auto end(std::size_t row) const noexcept { return col_indices.begin() + (row + 1) * elements_per_row; }
	auto cend(std::size_t row) const noexcept { return col_indices.cbegin() + (row + 1) * elements_per_row; }

	auto Row(std::size_t row) noexcept { return std::span(begin(row), end(row)); }
	auto Row(std::size_t row) const noexcept { return std::span(begin(row), end(row)); }
		
	// Returns diag(A' * A)
	template <typename T>
	std::vector<T> DiagATA() const
	{
		std::vector<T> ret(cols);
		#pragma omp parallel
		{
			std::vector<T> local_ret(cols); // prevents cache thrashing
			#pragma omp for nowait schedule(static)
			for (int64_t i = 0; i < static_cast<int64_t>(col_indices.size()); i++)
				local_ret[col_indices[i]] += 1;

			#pragma omp critical
			ret += local_ret;
		}
		return ret;
	}

	// Jacobi Preconditioner Square
	// Returns 1 / diag(A' * A)
	template <typename T>
	std::vector<T> JacobiPreconditionerSquare(T infinity = std::numeric_limits<T>::infinity()) const
	{
		auto diag = DiagATA<T>();
		for (auto& elem : diag)
			if (elem == 0)
				elem = infinity;
			else
				elem = T(1) / elem;
		return diag;
	}
};

template <typename T, typename U>
std::vector<T> operator*(const MatrixCSR<U>& mat, const std::vector<T>& x)
{
	if (x.size() != mat.Cols())
		throw std::runtime_error("Size mismatch.");

	const int64_t rows = static_cast<int64_t>(mat.Rows());
	std::vector<T> result(rows);
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < rows; i++)
		result[i] = std::accumulate(mat.begin(i), mat.end(i), T(0), [&x](T t, const auto& i) { return t + x[i]; });
	return result;
}

template <typename T, typename U>
std::vector<T> operator*(const std::vector<T>& x, const MatrixCSR<U>& mat)
{
	if (x.size() != mat.Rows())
		throw std::runtime_error("Size mismatch.");

	const int64_t rows = static_cast<int64_t>(mat.Rows());
	const int64_t cols = static_cast<int64_t>(mat.Cols());
	std::vector<T> result(cols);
	#pragma omp parallel
	{
		std::vector<T> local_result(cols); // prevents cache thrashing
		#pragma omp for nowait schedule(static)
		for (int64_t i = 0; i < rows; i++)
			for (auto col_index : mat.Row(i))
				local_result[col_index] += x[i];

		#pragma omp critical
		result += local_result;
	}
	return result;
}