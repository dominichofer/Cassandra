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

namespace
{
	template <typename IndexType>
	class TransposedProxy;

	template <typename IndexType>
	class ATA_Proxy;
}

// Compressed Sparse Row Matrix
// with fixed number of elements per row, with only 1 as element, but the matrix can have multiple entries per element.
template <typename IndexType>
class MatrixCSR : public IMatrix
{
	friend TransposedProxy<IndexType>;
public:
	using index_type = IndexType;
private:
	const std::size_t entires_per_row;
	const std::size_t cols;
	std::vector<index_type> col_indices;
public:
	MatrixCSR(std::size_t entires_per_row, std::size_t cols, std::size_t rows = 0) noexcept
		: entires_per_row(entires_per_row)
		, cols(cols)
		, col_indices(rows * entires_per_row)
	{
		if (std::numeric_limits<index_type>::max() < cols)
			throw std::runtime_error("Template type 'index_type' is too small to represent 'cols'.");
	}

	std::size_t Rows() const noexcept override { return col_indices.size() / entires_per_row; }
	std::size_t Cols() const noexcept override { return cols; }
	std::size_t size() const noexcept override { return col_indices.size(); }

	auto begin() noexcept { return col_indices.begin(); }
	auto begin() const noexcept { return col_indices.begin(); }
	auto cbegin() const noexcept { return col_indices.cbegin(); }
	auto end() noexcept { return col_indices.end(); }
	auto end() const noexcept { return col_indices.end(); }
	auto cend() const noexcept { return col_indices.cend(); }
	auto begin(std::size_t row) noexcept { return col_indices.begin() + row * entires_per_row; }
	auto begin(std::size_t row) const noexcept { return col_indices.begin() + row * entires_per_row; }
	auto cbegin(std::size_t row) const noexcept { return col_indices.cbegin() + row * entires_per_row; }
	auto end(std::size_t row) noexcept { return col_indices.begin() + (row + 1) * entires_per_row; }
	auto end(std::size_t row) const noexcept { return col_indices.begin() + (row + 1) * entires_per_row; }
	auto cend(std::size_t row) const noexcept { return col_indices.cbegin() + (row + 1) * entires_per_row; }

	// 'n'-th element in 'row'.
	index_type& operator()(std::size_t row, std::size_t n) noexcept { return col_indices[row * entires_per_row + n]; }

	// 'n'-th element in 'row'.
	const index_type& operator()(std::size_t row, std::size_t n) const noexcept { return col_indices[row * entires_per_row + n]; }

	Vector operator*(const Vector& x) const override
	{
		if(x.size() != cols)
			throw std::runtime_error("Size mismatch.");

		const int64_t rows = static_cast<int64_t>(Rows());
		Vector result(rows, 0);
		#pragma omp parallel for
		for (int64_t i = 0; i < rows; i++)
		{
			double sum = 0; // prevents cache thrashing
			for (auto j = i * entires_per_row; j < (i + 1) * entires_per_row; j++)
				sum += x[col_indices[j]];
			result[i] = sum;
		}
		return result;
	}

	Vector ATAx(const Vector& x) const override
	{
		if(x.size() != cols)
			throw std::runtime_error("Size mismatch.");

		const int64_t rows = static_cast<int64_t>(Rows());
		Vector result(cols, 0);
		#pragma omp parallel
		{
			Vector local_result(cols, 0); // prevents cache thrashing

			#pragma omp for nowait schedule(dynamic,64)
			for (int64_t i = 0; i < rows; i++)
			{
				double sum = 0; // prevents cache thrashing
				for (auto j = i * entires_per_row; j < (i + 1) * entires_per_row; j++)
					sum += x[col_indices[j]];
				for (auto j = i * entires_per_row; j < (i + 1) * entires_per_row; j++)
					local_result[col_indices[j]] += sum;
			}

			#pragma omp critical
			{
				for (std::size_t i = 0; i < cols; i++)
					result[i] += local_result[i];
			}
		}
		return result;
	}
	
	// Returns diag(A' * A)
	Vector DiagATA() const
	{
		const int64_t s = static_cast<int64_t>(size());
		Vector ret(cols, 0);
		#pragma omp parallel
		{
			Vector local_ret(cols, 0); // prevents cache thrashing

			#pragma omp for nowait
			for (int64_t i = 0; i < s; i++)
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
	Vector JacobiPreconditionerSquare(double infinity = std::numeric_limits<double>::infinity()) const { return inv(DiagATA(), infinity); }
};

namespace
{
	template <typename IndexType>
	class TransposedProxy final : public IMatrix
	{
		const MatrixCSR<IndexType>& o;
	public:
		TransposedProxy(const MatrixCSR<IndexType>& o) : o(o) {}

		std::size_t Rows() const noexcept override { return o.Cols(); }
		std::size_t Cols() const noexcept override { return o.Rows(); }
		std::size_t size() const noexcept override { return o.size(); }

		Vector operator*(const Vector& x) const override
		{
			if(x.size() != Cols())
				throw std::runtime_error("Size mismatch.");

			const int64_t cols = static_cast<int64_t>(Cols());
			Vector result(Rows(), 0);
			#pragma omp parallel
			{
				Vector local_result(Rows(), 0); // prevents cache thrashing

				#pragma omp for nowait
				for (int64_t i = 0; i < cols; i++)
					for (auto j = i * o.entires_per_row; j < (i + 1) * o.entires_per_row; j++)
						local_result[o.col_indices[j]] += x[i];

				#pragma omp critical
				{
					for (std::size_t i = 0; i < Rows(); i++)
						result[i] += local_result[i];
				}
			}
			return result;
		}
	};

	template <typename IndexType>
	class ATA_Proxy final : public IMatrix
	{
		const TransposedProxy<IndexType>& l;
		const MatrixCSR<IndexType>& r;
	public:
		ATA_Proxy(const TransposedProxy<IndexType>& l, const MatrixCSR<IndexType>& r)
			: l(l), r(r)
		{ assert(l.Cols() == r.Rows()); }

		std::size_t Rows() const noexcept override { return l.Rows(); }
		std::size_t Cols() const noexcept override { return r.Cols(); }
		std::size_t size() const noexcept override { return l.size() + r.size(); }

		Vector operator*(const Vector& x) const override { return l * (r * x); }
	};
}

template <typename IndexType>
TransposedProxy<IndexType> transposed(const MatrixCSR<IndexType>& m)
{
	return TransposedProxy<IndexType>(m);
}

template <typename IndexType>
ATA_Proxy<IndexType> operator*(const TransposedProxy<IndexType>& l, const MatrixCSR<IndexType>& r)
{
	return MM_Proxy(l, r);
}