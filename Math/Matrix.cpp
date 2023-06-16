#include "Matrix.h"
#include "Vector.h"
#include <stdexcept>

Matrix Matrix::operator=(const Matrix& o) noexcept
{
	cols = o.cols;
	data = o.data;
	return *this;
}

Matrix Matrix::operator=(Matrix&& o) noexcept
{
	cols = o.cols;
	data = std::move(o.data);
	return *this;
}

Matrix Matrix::Id(std::size_t size)
{
	Matrix m(size, size);
	for (std::size_t i = 0; i < size; i++)
		m(i, i) = 1;
	return m;
}

std::span<double> Matrix::Row(std::size_t index) noexcept
{
	return std::span(data.begin() + index * cols, data.begin() + (index + 1) * cols);
}

std::span<const double> Matrix::Row(std::size_t index) const noexcept
{
	return std::span(data.begin() + index * cols, data.begin() + (index + 1) * cols);
}

Matrix& Matrix::operator+=(const Matrix& o)
{
	if (Rows() != o.Rows() || Cols() != o.Cols())
		throw std::runtime_error("Size mismatch");

	const int64_t size = static_cast<int64_t>(data.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		data[i] += o.data[i];
	return *this;
}

Matrix& Matrix::operator-=(const Matrix& o)
{
	if (Rows() != o.Rows() || Cols() != o.Cols())
		throw std::runtime_error("Size mismatch");

	const int64_t size = static_cast<int64_t>(data.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		data[i] -= o.data[i];
	return *this;
}

Matrix& Matrix::operator*=(double factor)
{
	const int64_t size = static_cast<int64_t>(data.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		data[i] *= factor;
	return *this;
}

Matrix& Matrix::operator/=(double factor)
{
	const int64_t size = static_cast<int64_t>(data.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		data[i] /= factor;
	return *this;
}

Matrix operator+(Matrix l, const Matrix& r)
{
	return l += r;
}

Matrix operator+(const Matrix& l, Matrix&& r)
{
	return r += l;
}

Matrix operator-(Matrix l, const Matrix& r)
{
	return l -= r;
}

Matrix operator-(const Matrix& l, Matrix&& r)
{
	return r = l - r;
}

Matrix operator*(Matrix m, double factor)
{
	return m *= factor;
}

Matrix operator*(double factor, Matrix m)
{
	return m *= factor;
}

Matrix operator/(Matrix m, double factor)
{
	return m /= factor;
}

Vector operator*(const Matrix& mat, const Vector& x)
{
	std::size_t rows = mat.Rows();
	std::size_t cols = mat.Cols();
	if (x.size() != cols)
		throw std::runtime_error("Size mismatch");

	Vector result(rows, 0.0);
	#pragma omp parallel for schedule(static) if (cols * rows > 64 * 64)
	for (int64_t i = 0; i < static_cast<int64_t>(rows); i++)
	{
		double sum = 0;
		for (std::size_t j = 0; j < cols; j++)
			sum += mat(i, j) * x[j];
		result[i] = sum;
	}
	return result;
}

Vector operator*(const Vector& x, const Matrix& mat)
{
	std::size_t rows = mat.Rows();
	std::size_t cols = mat.Cols();
	if (x.size() != rows)
		throw std::runtime_error("Size mismatch");

	Vector result(cols, 0.0);
	if (cols * rows > 128 * 128)
	{
		#pragma omp parallel
		{
			Vector local_result(cols, 0.0); // prevents cache thrashing
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
				result[j] += x[i] * mat(i, j);
	}
	return result;
}

Matrix operator*(const Matrix& l, const Matrix& r)
{
	if (l.Cols() != r.Rows())
		throw std::runtime_error("Size mismatch");

	std::size_t rows = l.Rows();
	std::size_t cols = r.Cols();
	std::size_t K = l.Cols(); // contracted dimenstion
	Matrix ret(rows, cols);
	#pragma omp parallel for schedule(static) if (cols * rows > 128 * 128)
	for (int64_t i = 0; i < static_cast<int64_t>(rows); i++)
		for (std::size_t j = 0; j < cols; j++)
		{
			double sum = 0;
			for (std::size_t k = 0; k < K; k++)
				sum += l(i, k) * r(k, j);
			ret(i, j) = sum;
		}
	return ret;
}

Matrix transposed(const Matrix& mat)
{
	std::size_t rows = mat.Rows();
	std::size_t cols = mat.Cols();
	Matrix ret(cols, rows);
	#pragma omp parallel for schedule(static) if (cols * rows > 128 * 128)
	for (int64_t i = 0; i < static_cast<int64_t>(rows); i++)
		for (std::size_t j = 0; j < cols; j++)
			ret(j, i) = mat(i, j);
	return ret;
}

std::string to_string(const Matrix& m)
{
	std::string s = "{";
	for (std::size_t i = 0; i < m.Rows(); i++)
	{
		s += '{';
		for (std::size_t j = 0; j < m.Cols(); j++)
		{
			s += std::to_string(m(i, j));
			if (j < m.Cols() - 1)
				s += " ";
		}
		s += "}";
		if (i < m.Rows() - 1)
			s += "\n";
	}
	return s + '}';
}
