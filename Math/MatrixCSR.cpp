#include "MatrixCSR.h"
#include <algorithm>
#include <stdexcept>

MatrixCSR::MatrixCSR(std::size_t elements_per_row, std::size_t rows, std::size_t cols)
	: elements_per_row(elements_per_row)
	, cols(cols)
	, col_indices(rows* elements_per_row)
{}

std::span<uint32_t> MatrixCSR::Row(std::size_t row) noexcept
{
	return std::span(col_indices.begin() + row * elements_per_row, col_indices.begin() + (row + 1) * elements_per_row);
}

std::span<const uint32_t> MatrixCSR::Row(std::size_t row) const noexcept
{
	return std::span(col_indices.begin() + row * elements_per_row, col_indices.begin() + (row + 1) * elements_per_row);
}

Vector JacobiPreconditionerOfATA(const MatrixCSR& m)
{
	Vector diag_ATA(m.Cols(), 0.0);
	#pragma omp parallel
	{
		Vector local(m.Cols(), 0.0); // prevents cache thrashing
		#pragma omp for nowait schedule(static)
		for (int64_t row_index = 0; row_index < static_cast<int64_t>(m.Rows()); row_index++)
		{
			auto row = m.Row(row_index);
			std::vector<uint32_t> col_indices(row.begin(), row.end());
			std::ranges::sort(col_indices);
			int j;
			for  (int i = 0; i < col_indices.size(); i = j)
			{
				j = i + 1;
				while (j < col_indices.size() && col_indices[i] == col_indices[j])
					j++;
				local[col_indices[i]] += (j - i) * (j - i);
			}
		}

		#pragma omp critical
		diag_ATA += local;
	}
	return inv(diag_ATA);
}

Vector operator*(const MatrixCSR& mat, const Vector& x)
{
	if (x.size() != mat.Cols())
		throw std::runtime_error("Size mismatch.");

	const int64_t rows = static_cast<int64_t>(mat.Rows());
	Vector result(rows, 0.0);
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < rows; i++)
	{
		float sum = 0.0f;
		for (auto col_index : mat.Row(i))
			sum += x[col_index];
		result[i] = sum;
	}
	return result;
}

Vector operator*(const Vector& x, const MatrixCSR& mat)
{
	if (x.size() != mat.Rows())
		throw std::runtime_error("Size mismatch.");

	const int64_t rows = static_cast<int64_t>(mat.Rows());
	const int64_t cols = static_cast<int64_t>(mat.Cols());
	Vector result(cols, 0.0);
	#pragma omp parallel
	{
		Vector local_result(cols, 0.0); // prevents cache thrashing
		#pragma omp for nowait schedule(static)
		for (int64_t i = 0; i < rows; i++)
			for (auto col_index : mat.Row(i))
				local_result[col_index] += x[i];

		#pragma omp critical
		result += local_result;
	}
	return result;
}
