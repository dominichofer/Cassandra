#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <omp.h>
#include "Vector.h"
#include "MatrixInterface.h"

template <class ValueType>
class Matrix final : public IMatrix
{
public:
	using value_type = ValueType;
protected:
	const std::size_t rows, cols;
	std::vector<value_type> data;
public:
	Matrix(std::size_t rows, std::size_t cols) : rows(rows), cols(cols), data(rows * cols) {}

	// Identity matrix.
	static Matrix Id(std::size_t size)
	{
		Matrix m(size, size);
		for (std::size_t i = 0; i < size; i++)
			m(i, i) = 1;
		return m;
	}

	[[nodiscard]] auto operator==(const Matrix& o) const noexcept { return (row == o.rows) && (cols == o.cols) && (data == o.data); }
	[[nodiscard]] auto operator!=(const Matrix& o) const noexcept { return !(*this == o); }

	std::size_t Rows() const noexcept override { return rows; }
	std::size_t Cols() const noexcept override { return cols; }
	std::size_t size() const noexcept override { return data.size(); }

	value_type& operator()(std::size_t i, std::size_t j) { return data[i * cols + j]; }
	value_type const& operator()(std::size_t i, std::size_t j) const { return data[i * cols + j]; }

	Vector operator*(const Vector& x) const override
	{
		assert(x.size() == cols);

		Vector result(rows);
		#pragma omp parallel for
		for (int64_t i = 0; i < static_cast<int64_t>(rows); i++)
		{
			double sum = 0; // prevents cache thrashing
			for (std::size_t j = 0; j < cols; j++)
				sum += this->operator()(i, j) * x[j];
			result[i] = static_cast<Vector::value_type>(sum);
		}
		return result;
	}

	Matrix<value_type>& operator+=(const Matrix<value_type>& o) 
	{
		const int64_t size = data.size();
		#pragma omp parallel for
		for (int64_t i = 0; i < size; i++)
			data[i] += o.data[i];
		return *this;
	}

	Matrix<value_type>& operator-=(const Matrix<value_type>& o)
	{
		const int64_t size = data.size();
		#pragma omp parallel for
		for (int64_t i = 0; i < size; i++)
			data[i] -= o.data[i];
		return *this;
	}

	template <class T>
	Matrix<value_type>& operator*=(const T& o) 
	{
		const int64_t size = data.size();
		#pragma omp parallel for
		for (int64_t i = 0; i < size; i++)
			data[i] *= o;
		return *this;
	}

	template <class T>
	Matrix<value_type>& operator/=(const T& o) 
	{
		const int64_t size = data.size();
		#pragma omp parallel for
		for (int64_t i = 0; i < size; i++)
			data[i] /= o;
		return *this;
	}

	Vector ATx(const Vector& x) const
	{
		assert(x.size() == rows);

		Vector result(cols);
		#pragma omp parallel
		{
			Vector local_result(cols, 0); // prevents cache thrashing

			#pragma omp for nowait
			for (int64_t i = 0; i < static_cast<int64_t>(rows); i++)
				for (std::size_t j = 0; j < cols; j++)
					local_result[j] += this->operator()(i, j) * x[i];

			#pragma omp critical
			{
				for (std::size_t i = 0; i < cols; i++)
					result[i] += local_result[i];
			}
		}
		return result;
	}
};

template <typename T> Matrix<T> operator+(      Matrix<T>  a, const Matrix<T>&  b) { return a += b; }
template <typename T> Matrix<T> operator+(const Matrix<T>& a,       Matrix<T>&& b) { return b += a; }

template <typename T> Matrix<T> operator-(      Matrix<T>  a, const Matrix<T>&  b) { return a -= b; }
template <typename T> Matrix<T> operator-(const Matrix<T>& l,       Matrix<T>&& r)
{
	assert(l.size() == r.size());
	const int64_t size = r.size();
	#pragma omp parallel for
	for (int64_t i = 0; i < size; i++)
		r[i] = l[i] - r[i];
	return r;
}

template <typename T, typename U> Matrix<T> operator*(Matrix<T> mat, const U& mul) { return mat *= mul; }
template <typename T, typename U> Matrix<T> operator*(const U& mul, Matrix<T> mat) { return mat *= mul; }

template <typename T, typename U> Matrix<T> operator/(Matrix<T> a, const U& b) { return a /= b; }


template <typename T>
Matrix<T> Transposed(const Matrix<T>& in)
{
	Matrix<T> out(in.Cols(), in.Rows());
	for (std::size_t i = 0; i < in.Rows(); i++)
		for (std::size_t j = 0; j < in.Cols(); j++)
			out(j, i) = in(i, j);
	return out;
}
