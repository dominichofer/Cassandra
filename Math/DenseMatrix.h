#pragma once
#include <string>
#include <cstdint>
#include <cstddef>
#include <ranges>
#include <stdexcept>
#include <vector>
#include <omp.h>
#include "Vector.h"

template <typename ValueType>
class DenseMatrix
{
public:
	using value_type = ValueType;
private:
	std::size_t cols;
	std::vector<value_type> data;
public:
	DenseMatrix(std::size_t rows, std::size_t cols) : cols(cols), data(rows * cols) {}

	// Identity matrix
	static DenseMatrix Id(std::size_t size)
	{
		DenseMatrix m(size, size);
		for (std::size_t i = 0; i < size; i++)
			m(i, i) = 1;
		return m;
	}

	bool operator==(const DenseMatrix& o) const noexcept { return (cols == o.cols) and (data == o.data); }
	bool operator!=(const DenseMatrix& o) const noexcept { return !(*this == o); }

	std::size_t Rows() const noexcept { return data.size() / cols; }
	std::size_t Cols() const noexcept { return cols; }
	
	      value_type& operator()(std::size_t row, std::size_t col)       { return data[row * cols + col]; }
	const value_type& operator()(std::size_t row, std::size_t col) const { return data[row * cols + col]; }

	auto Row(std::size_t i)       { return std::ranges::subrange(data.begin() + i * cols, data.begin() + (i + 1) * cols); }
	auto Row(std::size_t i) const { return std::ranges::subrange(data.begin() + i * cols, data.begin() + (i + 1) * cols); }
	//auto Col(std::size_t i)       { return data[std::slice(i, Rows(), cols)]; }
	//auto Col(std::size_t i) const { return data[std::slice(i, Rows(), cols)]; }

	DenseMatrix& operator+=(const DenseMatrix& o)
	{
		if (Rows() != o.Rows() || Cols() != o.Cols())
			throw std::runtime_error("Size mismatch");

		data += o.data;
		return *this;
	}

	DenseMatrix& operator-=(const DenseMatrix& o)
	{
		if (Rows() != o.Rows() || Cols() != o.Cols())
			throw std::runtime_error("Size mismatch");

		data -= o.data;
		return *this;
	}

	DenseMatrix& operator*=(const value_type& m) 
	{
		data *= m;
		return *this;
	}

	DenseMatrix& operator/=(const value_type& m) 
	{
		data /= m;
		return *this;
	}
};


template <typename T> DenseMatrix<T> operator+(      DenseMatrix<T>  l, const DenseMatrix<T>&  r) { return l += r; }
template <typename T> DenseMatrix<T> operator+(const DenseMatrix<T>& l,       DenseMatrix<T>&& r) { return r += l; }

template <typename T> DenseMatrix<T> operator-(      DenseMatrix<T>  l, const DenseMatrix<T>&  r) { return l -= r; }
template <typename T> DenseMatrix<T> operator-(const DenseMatrix<T>& l,       DenseMatrix<T>&& r) { return r = l - r; }

template <typename T> DenseMatrix<T> operator*(DenseMatrix<T> mat, const T& mul) { return mat *= mul; }
template <typename T> DenseMatrix<T> operator*(const T& mul, DenseMatrix<T> mat) { return mat *= mul; }

template <typename T> DenseMatrix<T> operator/(DenseMatrix<T> mat, const T& mul) { return mat /= mul; }

template <typename T, typename U>
std::vector<U> operator*(const DenseMatrix<T>& mat, const std::vector<U>& x)
{
	auto rows = mat.Rows();
	auto cols = mat.Cols();
	if (x.size() != cols)
		throw std::runtime_error("Size mismatch");

	std::vector<U> result(rows);
	#pragma omp parallel for schedule(static) if (cols * rows > 64 * 64)
	for (int64_t i = 0; i < static_cast<int64_t>(rows); i++)
	{
		U sum = 0;
		for (std::size_t j = 0; j < cols; j++)
			sum += mat(i, j) * x[j];
		result[i] = sum;
	}
	return result;
}

template <typename T, typename U>
std::vector<U> operator*(const std::vector<U>& x, const DenseMatrix<T>& mat)
{
	auto rows = mat.Rows();
	auto cols = mat.Cols();
	if (x.size() != rows)
		throw std::runtime_error("Size mismatch");

	std::vector<U> result(cols);
	if (cols * rows > 128 * 128)
	{
		#pragma omp parallel
		{
			std::vector<U> local_result(cols); // prevents cache thrashing
			#pragma omp for nowait schedule(static)
			for (int64_t i = 0; i < static_cast<int64_t>(rows); i++)
				for (std::size_t j = 0; j < cols; j++)
					local_result[j] += x[i] * mat(i, j);

			#pragma omp critical
			result += local_result;
		}
	}
	else
	{
		for (int64_t i = 0; i < static_cast<int64_t>(rows); i++)
			for (std::size_t j = 0; j < cols; j++)
				result[j] += mat(i, j) * x[i];
	}
	return result;
}

template <typename T>
DenseMatrix<T> operator*(const DenseMatrix<T>& l, const DenseMatrix<T>& r)
{
	if (l.Cols() != r.Rows())
		throw std::runtime_error("Size mismatch");

	auto rows = l.Rows();
	auto cols = r.Cols();
	auto K = l.Cols(); // contracted dimenstion
	DenseMatrix<T> ret(rows, cols);
	#pragma omp parallel for schedule(static) if (cols * rows > 128 * 128)
	for (int64_t i = 0; i < static_cast<int64_t>(rows); i++)
		for (std::size_t j = 0; j < cols; j++)
		{
			T sum = 0;
			for (std::size_t k = 0; k < K; k++)
				sum += l(i, k) * r(k, j);
			ret(i, j) = sum;
		}
	return ret;
}

template <typename T>
DenseMatrix<T> transposed(const DenseMatrix<T>& mat)
{
	auto rows = mat.Rows();
	auto cols = mat.Cols();
	DenseMatrix<T> ret(cols, rows);
	#pragma omp parallel for schedule(static) if (cols * rows > 128 * 128)
	for (int64_t i = 0; i < static_cast<int64_t>(rows); i++)
		for (std::size_t j = 0; j < cols; j++)
			ret(j, i) = mat(i, j);
	return ret;
}

template <typename T>
std::string to_string(const DenseMatrix<T>& m)
{
	using std::to_string;
	std::string s = "{";
	for (std::size_t i = 0; i < m.Rows(); i++)
	{
		s += '{';
		for (std::size_t j = 0; j < m.Cols(); j++)
			s += to_string(m(i, j)) + ", ";
		s += "},\n";
	}
	return s + '}';
}
