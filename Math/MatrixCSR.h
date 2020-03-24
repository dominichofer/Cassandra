#pragma once
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <numeric>
#include <vector>
#include "Vector.h"
#include "MatrixInterface.h"

namespace
{
	template <typename ValueType, typename IndexType>
	class TransposedProxy;
}

// Compressed Sparse Row Matrix
template <typename ValueType, typename IndexType = std::size_t>
class MatrixCSR : public IMatrix
{
	friend TransposedProxy<ValueType, IndexType>;
public:
	using value_type = ValueType;
	using index_type = IndexType;
private:
	const std::size_t cols;

	std::vector<index_type> row_starts;
	std::vector<index_type> col_indices;
	std::vector<value_type> data;

	bool Constrained() const { return (data.size() == col_indices.size()) && std::is_sorted(row_starts.begin(), row_starts.end()); }

public:
	MatrixCSR(std::size_t cols = 0) noexcept : cols(cols), row_starts(1, 0) {}
	
	[[nodiscard]] auto operator==(const MatrixCSR&) const noexcept { return (cols == o.cols) && (row_starts == o.row_starts) && (col_indices == o.col_indices) && (data == o.data); }
	[[nodiscard]] auto operator!=(const MatrixCSR&) const noexcept { return !(data == o.data); }

	std::size_t Rows() const noexcept override { return row_starts.size() - 1; }
	std::size_t Cols() const noexcept override { return cols; }
	std::size_t size() const noexcept override { return col_indices.size(); }
	std::size_t nnz() const noexcept override { return std::count_if(data.begin(), data.end(), [](const auto& d) { return d != 0; }); }
	
	void push_back(index_type col, value_type datum) {
		assert(col < cols);
		col_indices.emplace_back(std::move(col));
		data.emplace_back(std::move(datum));
	}

	void end_row() { row_starts.push_back(size()); }

	Vector operator*(const Vector& x) const override
	{
		assert(x.size() == cols);

		const int64_t rows = static_cast<int64_t>(Rows());
		Vector result(rows, 0);
		#pragma omp parallel for
		for (int64_t i = 0; i < rows; i++)
		{
			double sum = 0; // prevents cache thrashing
			for (auto j = row_starts[i]; j < row_starts[i + 1]; j++)
				sum += data[j] * x[col_indices[j]];
			result[i] = sum;
		}
		return result;
	}

	Vector ATAx(const Vector& x) const override
	{
		//return ATx(Ax(x)); // Room for optimization: Remove this!

		assert(x.size() == cols);

		const int64_t rows = static_cast<int64_t>(Rows());
		Vector result(cols, 0);
		#pragma omp parallel
		{
			Vector local_result(cols, 0); // prevents cache thrashing

			#pragma omp for nowait schedule(dynamic,64)
			for (int64_t i = 0; i < rows; i++)
			{
				double sum = 0; // prevents cache thrashing
				for (auto j = row_starts[i]; j < row_starts[i + 1]; j++)
					sum += data[j] * x[col_indices[j]];
				for (auto j = row_starts[i]; j < row_starts[i + 1]; j++)
					local_result[col_indices[j]] += data[j] * sum;
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
				local_ret[col_indices[i]] += static_cast<double>(data[i]) * static_cast<double>(data[i]);

			#pragma omp critical
			{
				ret += local_ret;
			}
		}
		return ret;
	}

	// Jacobi Preconditioner Square
	// Returns 1 / diag(A' * A)
	Vector JacobiPreconditionerSquare() const { return inv(DiagATA(), 1000); }
};

namespace
{
	template <typename ValueType, typename IndexType>
	class TransposedProxy final : public IMatrix
	{
		const MatrixCSR<ValueType, IndexType>& o;
	public:
		TransposedProxy(const MatrixCSR<ValueType, IndexType>& o) : o(o) {}

		std::size_t Rows() const noexcept override { return o.Cols(); }
		std::size_t Cols() const noexcept override { return o.Rows(); }
		std::size_t size() const noexcept override { return o.size(); }
		std::size_t nnz() const noexcept override { return o.nnz(); }

		Vector operator*(const Vector& x) const override
		{
			auto x_size = x.size();
			auto col = Cols();
			assert(x.size() == Cols());

			const int64_t cols = static_cast<int64_t>(Cols());
			Vector result(Rows(), 0);
			#pragma omp parallel
			{
				Vector local_result(Rows(), 0); // prevents cache thrashing

				#pragma omp for nowait
				for (int64_t i = 0; i < cols; i++)
					for (auto j = o.row_starts[i]; j < o.row_starts[i + 1]; j++)
						local_result[o.col_indices[j]] += o.data[j] * x[i];

				#pragma omp critical
				{
					for (std::size_t i = 0; i < Rows(); i++)
						result[i] += local_result[i];
				}
			}
			return result;
		}
	};

	class MM_Proxy final : public IMatrix
	{
		const IMatrix& l;
		const IMatrix& r;
	public:
		MM_Proxy(const IMatrix& l, const IMatrix& r) : l(l), r(r) { assert(l.Cols() == r.Rows()); }

		std::size_t Rows() const noexcept override { return l.Rows(); }
		std::size_t Cols() const noexcept override { return r.Cols(); }
		std::size_t size() const noexcept override { return l.size() + r.size(); }
		std::size_t nnz() const noexcept override { return 0; }

		Vector operator*(const Vector& x) const override { return l * (r * x); }
	};
}

template <typename ValueType, typename IndexType>
TransposedProxy<ValueType, IndexType> transposed(const MatrixCSR<ValueType, IndexType>& m)
{
	return TransposedProxy<ValueType, IndexType>(m);
}

inline MM_Proxy operator*(const IMatrix& l, const IMatrix& r) { return MM_Proxy(l, r); }