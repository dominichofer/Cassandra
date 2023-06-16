#pragma once
#include <cstddef>
#include <span>
#include <string>
#include <vector>

// Forward declaration
class Vector;

class Matrix
{
	std::size_t cols;
	std::vector<double> data;
public:
	Matrix(std::size_t rows, std::size_t cols) noexcept : cols(cols), data(rows * cols, 0.0) {}

	Matrix(const Matrix& o) noexcept : cols(o.cols), data(o.data) {}
	Matrix(Matrix&& o) noexcept : cols(o.cols), data(std::move(o.data)) {}

	Matrix operator=(const Matrix&) noexcept;
	Matrix operator=(Matrix&&) noexcept;

	// Identity matrix
	static Matrix Id(std::size_t size);

	bool operator==(const Matrix& o) const noexcept { return (cols == o.cols) and (data == o.data); }
	bool operator!=(const Matrix& o) const noexcept { return !(*this == o); }

	std::size_t Rows() const noexcept { return data.size() / cols; }
	std::size_t Cols() const noexcept { return cols; }

	std::span<      double> Row(std::size_t index)       noexcept;
	std::span<const double> Row(std::size_t index) const noexcept;

	      double& operator()(std::size_t row, std::size_t col)       { return data[row * cols + col]; }
	const double& operator()(std::size_t row, std::size_t col) const { return data[row * cols + col]; }

	Matrix& operator+=(const Matrix&);
	Matrix& operator-=(const Matrix&);
	Matrix& operator*=(double);
	Matrix& operator/=(double);
};

Matrix operator+(Matrix, const Matrix&);
Matrix operator+(const Matrix&, Matrix&&);
Matrix operator-(Matrix, const Matrix&);
Matrix operator-(const Matrix&, Matrix&&);
Matrix operator*(Matrix, double);
Matrix operator*(double, Matrix);
Matrix operator/(Matrix, double);

Vector operator*(const Matrix&, const Vector&);
Vector operator*(const Vector&, const Matrix&);

Matrix operator*(const Matrix&, const Matrix&);

Matrix transposed(const Matrix&);
std::string to_string(const Matrix&);
