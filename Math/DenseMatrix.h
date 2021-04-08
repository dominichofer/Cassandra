#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <omp.h>
#include "Vector.h"
#include "Matrix.h"

template <class ValueType>
class DenseMatrix final : public Matrix
{
public:
	using value_type = ValueType;
	const std::size_t rows, cols;
	std::vector<value_type> data;
public:
	DenseMatrix(std::size_t rows, std::size_t cols) : rows(rows), cols(cols), data(rows * cols) {}

	// Identity matrix.
	static DenseMatrix Id(std::size_t size)
	{
		DenseMatrix m(size, size);
		for (std::size_t i = 0; i < size; i++)
			m(i, i) = 1;
		return m;
	}

	[[nodiscard]] auto operator==(const DenseMatrix& o) const noexcept { return (row == o.rows) && (cols == o.cols) && (data == o.data); }
	[[nodiscard]] auto operator!=(const DenseMatrix& o) const noexcept { return !(*this == o); }

	std::size_t Rows() const noexcept override { return rows; }
	std::size_t Cols() const noexcept override { return cols; }

	Vector Ax(const Vector& x) const override
	{
		assert(x.size() == cols);

		Vector result(rows);
		#pragma omp parallel for schedule(static)
		for (int64_t i = 0; i < static_cast<int64_t>(rows); i++)
		{
			Vector::value_type sum = 0; // prevents cache thrashing
			for (std::size_t j = 0; j < cols; j++)
				sum += data[i * cols + j] * x[j];
			result[i] = sum;
		}
		return result;
	}

	Vector ATx(const Vector& x) const override
	{
		assert(x.size() == rows);

		Vector result(cols);
		#pragma omp parallel
		{
			Vector local_result(cols, 0); // prevents cache thrashing

			#pragma omp for nowait schedule(static)
			for (int64_t i = 0; i < static_cast<int64_t>(rows); i++)
				for (std::size_t j = 0; j < cols; j++)
					local_result[j] += data[i * cols + j] * x[i];

			#pragma omp critical
			{
				result += local_result;
			}
		}
		return result;
	}

	//Vector ATAx(const Vector& x) const override { return ATx(Ax(x)); }

	      value_type& operator()(std::size_t i, std::size_t j)       { return data[i * cols + j]; }
	const value_type& operator()(std::size_t i, std::size_t j) const { return data[i * cols + j]; }

	DenseMatrix<value_type>& operator+=(const DenseMatrix<value_type>& o) 
	{
		if (Rows() != o.Rows() || Cols() != o.Cols())
			throw std::runtime_error("Size mismatch.");

		const int64_t size = static_cast<int64_t>(data.size());
		#pragma omp parallel for schedule(static)
		for (int64_t i = 0; i < size; i++)
			data[i] += o.data[i];
		return *this;
	}

	DenseMatrix<value_type>& operator-=(const DenseMatrix<value_type>& o)
	{
		if (Rows() != o.Rows() || Cols() != o.Cols())
			throw std::runtime_error("Size mismatch.");

		const int64_t size = static_cast<int64_t>(data.size());
		#pragma omp parallel for schedule(static)
		for (int64_t i = 0; i < size; i++)
			data[i] -= o.data[i];
		return *this;
	}

	DenseMatrix<value_type>& operator*=(const value_type& m) 
	{
		const int64_t size = static_cast<int64_t>(data.size());
		#pragma omp parallel for schedule(static)
		for (int64_t i = 0; i < size; i++)
			data[i] *= m;
		return *this;
	}

	DenseMatrix<value_type>& operator/=(const value_type& m) 
	{
		const int64_t size = static_cast<int64_t>(data.size());
		#pragma omp parallel for schedule(static)
		for (int64_t i = 0; i < size; i++)
			data[i] /= m;
		return *this;
	}
};

template <typename T> DenseMatrix<T> operator+(      DenseMatrix<T>  a, const DenseMatrix<T>&  b) { return a += b; }
template <typename T> DenseMatrix<T> operator+(const DenseMatrix<T>& a,       DenseMatrix<T>&& b) { return b += a; }

template <typename T> DenseMatrix<T> operator-(      DenseMatrix<T>  a, const DenseMatrix<T>&  b) { return a -= b; }
template <typename T> DenseMatrix<T> operator-(const DenseMatrix<T>& l,       DenseMatrix<T>&& r)
{
	assert(l.size() == r.size());
	const int64_t size = static_cast<int64_t>(r.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		r[i] = l[i] - r[i];
	return r;
}

template <typename T> DenseMatrix<T> operator*(const DenseMatrix<T>& l, const DenseMatrix<T>& r)
{
	assert(l.Cols() == r.Rows());
	DenseMatrix<T> ret(l.Rows(), r.Cols());
	for (int i = 0; i < ret.Rows(); i++)
		for (int j = 0; j < ret.Cols(); j++)
			for (int k = 0; k < l.Cols(); k++)
				ret(i,j) += l(i,k) * r(k,j);
	return ret;
}
template <typename T> DenseMatrix<T> operator*(DenseMatrix<T> mat, const T& mul) { return mat *= mul; }
template <typename T> DenseMatrix<T> operator*(const T& mul, DenseMatrix<T> mat) { return mat *= mul; }

template <typename T> DenseMatrix<T> operator/(DenseMatrix<T> a, const T& b) { return a /= b; }


template <typename T>
DenseMatrix<T> transposed(const DenseMatrix<T>& in)
{
	DenseMatrix<T> out(in.Cols(), in.Rows());
	for (std::size_t i = 0; i < in.Rows(); i++)
		for (std::size_t j = 0; j < in.Cols(); j++)
			out(j, i) = in(i, j);
	return out;
}
