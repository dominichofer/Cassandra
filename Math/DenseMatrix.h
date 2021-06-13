#pragma once
#include <string>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <omp.h>
#include "Vector.h"
#include "Matrix.h"

class DenseMatrix final : public Matrix
{
public:
	using value_type = double;
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

	[[nodiscard]] auto operator==(const DenseMatrix& o) const noexcept { return (rows == o.rows) && (cols == o.cols) && (data == o.data); }
	[[nodiscard]] auto operator!=(const DenseMatrix& o) const noexcept { return !(*this == o); }

	std::size_t Rows() const noexcept override { return rows; }
	std::size_t Cols() const noexcept override { return cols; }
	
	      value_type& operator()(std::size_t i, std::size_t j)       { return data[i * cols + j]; }
	const value_type& operator()(std::size_t i, std::size_t j) const { return data[i * cols + j]; }

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

	DenseMatrix& operator+=(const DenseMatrix& o) 
	{
		if (Rows() != o.Rows() || Cols() != o.Cols())
			throw std::runtime_error("Size mismatch.");

		const int64_t size = static_cast<int64_t>(data.size());
		#pragma omp parallel for schedule(static)
		for (int64_t i = 0; i < size; i++)
			data[i] += o.data[i];
		return *this;
	}

	DenseMatrix& operator-=(const DenseMatrix& o)
	{
		if (Rows() != o.Rows() || Cols() != o.Cols())
			throw std::runtime_error("Size mismatch.");

		const int64_t size = static_cast<int64_t>(data.size());
		#pragma omp parallel for schedule(static)
		for (int64_t i = 0; i < size; i++)
			data[i] -= o.data[i];
		return *this;
	}

	DenseMatrix& operator*=(const value_type& m) 
	{
		const int64_t size = static_cast<int64_t>(data.size());
		#pragma omp parallel for schedule(static)
		for (int64_t i = 0; i < size; i++)
			data[i] *= m;
		return *this;
	}

	DenseMatrix& operator/=(const value_type& m) 
	{
		const int64_t size = static_cast<int64_t>(data.size());
		#pragma omp parallel for schedule(static)
		for (int64_t i = 0; i < size; i++)
			data[i] /= m;
		return *this;
	}
};

inline DenseMatrix operator+(      DenseMatrix  a, const DenseMatrix&  b) { return a += b; }
inline DenseMatrix operator+(const DenseMatrix& a,       DenseMatrix&& b) { return b += a; }

inline DenseMatrix operator-(      DenseMatrix  a, const DenseMatrix&  b) { return a -= b; }
inline DenseMatrix operator-(const DenseMatrix& l,       DenseMatrix&& r)
{
	assert(l.Cols() == r.Cols());
	assert(l.Rows() == r.Rows());
	for (int i = 0; i < l.Rows(); i++)
		for (int j = 0; j < l.Cols(); j++)
			r(i,j) += l(i,j) - r(i,j);
	return r;
}

inline DenseMatrix operator*(const DenseMatrix& l, const DenseMatrix& r)
{
	assert(l.Cols() == r.Rows());
	DenseMatrix ret(l.Rows(), r.Cols());
	for (int i = 0; i < ret.Rows(); i++)
		for (int j = 0; j < ret.Cols(); j++)
			for (int k = 0; k < l.Cols(); k++)
				ret(i,j) += l(i,k) * r(k,j);
	return ret;
}
inline DenseMatrix operator*(DenseMatrix mat, DenseMatrix::value_type mul) { return mat *= mul; }
inline DenseMatrix operator*(DenseMatrix::value_type mul, DenseMatrix mat) { return mat *= mul; }

inline DenseMatrix operator/(DenseMatrix a, DenseMatrix::value_type b) { return a /= b; }

inline std::string to_string(const DenseMatrix& m)
{
	std::string s = "{";
	for (std::size_t i = 0; i < m.Rows(); i++)
	{
		s += '{';
		for (std::size_t j = 0; j < m.Cols(); j++)
			s += std::to_string(m(i, j)) + ", ";
		s += "},\n";
	}
	return s + '}';
}

inline DenseMatrix transposed(const DenseMatrix& in)
{
	DenseMatrix out(in.Cols(), in.Rows());
	for (std::size_t i = 0; i < in.Rows(); i++)
		for (std::size_t j = 0; j < in.Cols(); j++)
			out(j, i) = in(i, j);
	return out;
}
