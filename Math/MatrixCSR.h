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
	class ATB_Proxy;
}

// Compressed Sparse Row Matrix
// with fixed number of elements per row, with only 1 as element, but the matrix can have multiple entries per element.
template <typename IndexType>
class MatrixCSR : public IMatrix
{
	friend TransposedProxy<IndexType>;
	friend ATB_Proxy<IndexType>;
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

	Vector operator*(const Vector& x) const override
	{
		if(x.size() != cols)
			throw std::runtime_error("Size mismatch.");
		if(entires_per_row % 4 != 0)
			throw std::runtime_error("entires_per_row must be equal to 0 (mod 4).");

		const int64_t rows = static_cast<int64_t>(Rows());
		Vector result(rows, 0);
		#pragma omp parallel for
		for (int64_t i = 0; i < rows; i++)
		{
			double sum = 0; // prevents cache thrashing
			for (auto j = i * entires_per_row; j < (i + 1) * entires_per_row; j+=4)
				sum += x[col_indices[j+0]] + x[col_indices[j+1]] + x[col_indices[j+2]] + x[col_indices[j+3]];
			result[i] = sum;
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
	Vector JacobiPreconditionerSquare(double infinity = std::numeric_limits<double>::infinity()) const
	{
		return inv(DiagATA(), infinity);
	}
};

namespace
{
	template <typename IndexType>
	class TransposedProxy final : public IMatrix
	{
		const MatrixCSR<IndexType>& m;
	public:
		TransposedProxy(const MatrixCSR<IndexType>& m) : m(m) {}

		std::size_t Rows() const noexcept override { return m.Cols(); }
		std::size_t Cols() const noexcept override { return m.Rows(); }
		std::size_t size() const noexcept override { return m.size(); }

		auto begin() noexcept { return m.begin(); }
		auto begin() const noexcept { return m.begin(); }
		auto cbegin() const noexcept { return m.cbegin(); }
		auto end() noexcept { return m.end(); }
		auto end() const noexcept { return m.end(); }
		auto cend() const noexcept { return m.cend(); }
		auto begin(std::size_t row) noexcept { return m.begin(row); }
		auto begin(std::size_t row) const noexcept { return m.begin(row); }
		auto cbegin(std::size_t row) const noexcept { return m.cbegin(row); }
		auto end(std::size_t row) noexcept { return m.begin(row); }
		auto end(std::size_t row) const noexcept { return m.begin(row); }
		auto cend(std::size_t row) const noexcept { return m.cbegin(row); }

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
					for (auto j = i * m.entires_per_row; j < (i + 1) * m.entires_per_row; j++)
						local_result[m.col_indices[j]] += x[i];

				#pragma omp critical
				{
					result += local_result;
				}
			}
			return result;
		}
	};

	template <typename IndexType>
	class ATB_Proxy final : public IMatrix
	{
		const TransposedProxy<IndexType>& l;
		const MatrixCSR<IndexType>& r;
	public:
		ATB_Proxy(const TransposedProxy<IndexType>& l, const MatrixCSR<IndexType>& r)
			: l(l), r(r)
		{ assert(l.Cols() == r.Rows()); }

		std::size_t Rows() const noexcept override { return l.Rows(); }
		std::size_t Cols() const noexcept override { return r.Cols(); }
		std::size_t size() const noexcept override { return l.size() + r.size(); }

		Vector operator*(const Vector& x) const override
		{
			if(x.size() != r.cols)
				throw std::runtime_error("Size mismatch.");

			if (l.begin() != r.begin())
				return l * (r * x);

			if(r.entires_per_row % 4 != 0)
				throw std::runtime_error("entires_per_row must be equal to 0 (mod 4).");

			const int64_t rows = static_cast<int64_t>(r.Rows());
			Vector result(r.cols, 0);
			#pragma omp parallel
			{
				Vector local_result(r.cols, 0); // prevents cache thrashing

				#pragma omp for nowait schedule(dynamic,64)
				for (int64_t i = 0; i < rows; i++)
				{
					double sum = 0;
					for (auto j = i * r.entires_per_row; j < (i + 1) * r.entires_per_row; j+=4)
						sum += x[r.col_indices[j+0]] + x[r.col_indices[j+1]] + x[r.col_indices[j+2]] + x[r.col_indices[j+3]];
					for (auto j = i * r.entires_per_row; j < (i + 1) * r.entires_per_row; j+=4) {
						local_result[r.col_indices[j+0]] += sum;
						local_result[r.col_indices[j+1]] += sum;
						local_result[r.col_indices[j+2]] += sum;
						local_result[r.col_indices[j+3]] += sum;
					}
				}

				#pragma omp critical
				{
					result += local_result;
				}
			}
			return result;
		}
	};
}

template <typename IndexType>
auto transposed(const MatrixCSR<IndexType>& m)
{
	return TransposedProxy<IndexType>(m);
}

template <typename IndexType>
auto operator*(const TransposedProxy<IndexType>& l, const MatrixCSR<IndexType>& r)
{
	return ATB_Proxy(l, r);
}