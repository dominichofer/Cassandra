#pragma once
#include <cstdint>
#include <stdexcept>
#include "Vector.h"

// Interface
class Matrix
{
public:
	virtual std::size_t Rows() const noexcept = 0;
	virtual std::size_t Cols() const noexcept = 0;

	virtual Vector Ax(const Vector&) const = 0;
	virtual Vector ATx(const Vector&) const = 0;
	//virtual Vector ATAx(const Vector&) const = 0;
};

namespace
{
	// Proxy
	class TransposedMatrix final : public Matrix
	{
		const Matrix& m;
	public:
		explicit TransposedMatrix(const Matrix& m) noexcept : m(m) {}

		std::size_t Rows() const noexcept override { return m.Cols(); }
		std::size_t Cols() const noexcept override { return m.Rows(); }

		Vector Ax(const Vector& x) const override { return m.ATx(x); }
		Vector ATx(const Vector& x) const override { return m.Ax(x); }
		//Vector ATAx(const Vector& x) const override { return m.ATAx(x); }

		//auto begin() noexcept { return m.begin(); }
		//auto begin() const noexcept { return m.begin(); }
		//auto cbegin() const noexcept { return m.cbegin(); }
		//auto end() noexcept { return m.end(); }
		//auto end() const noexcept { return m.end(); }
		//auto cend() const noexcept { return m.cend(); }

		//auto begin(std::size_t row) noexcept { return m.begin(row); }
		//auto begin(std::size_t row) const noexcept { return m.begin(row); }
		//auto cbegin(std::size_t row) const noexcept { return m.cbegin(row); }
		//auto end(std::size_t row) noexcept { return m.begin(row); }
		//auto end(std::size_t row) const noexcept { return m.begin(row); }
		//auto cend(std::size_t row) const noexcept { return m.cbegin(row); }
	};

	// Proxy
	class TwoMatrix final : public Matrix
	{
		const Matrix& l;
		const Matrix& r;
	public:
		TwoMatrix(const Matrix& l, const Matrix& r) : l(l), r(r)
		{
			if(l.Cols() != r.Rows())
				throw std::runtime_error("Size mismatch.");
		}

		std::size_t Rows() const noexcept override { return l.Rows(); }
		std::size_t Cols() const noexcept override { return r.Cols(); }

		Vector Ax(const Vector& x) const override { return l.Ax(r.Ax(x)); }
		Vector ATx(const Vector& x) const override { return r.ATx(l.ATx(x)); }
		//Vector ATAx(const Vector& x) const override { return ATx(Ax(x)); }

		//Vector operator*(const Vector& x) const override
		//{
		//	if(x.size() != r.cols)
		//		throw std::runtime_error("Size mismatch.");

		//	if (l.begin() != r.begin())
		//		return l * (r * x);

		//	if(r.row_size % 4 != 0)
		//		throw std::runtime_error("entires_per_row must be equal to 0 (mod 4).");

		//	const int64_t rows = static_cast<int64_t>(r.Rows());
		//	Vector result(r.cols, 0);
		//	#pragma omp parallel
		//	{
		//		Vector local_result(r.cols, 0); // prevents cache thrashing

		//		#pragma omp for nowait schedule(dynamic,64)
		//		for (int64_t i = 0; i < rows; i++)
		//		{
		//			const auto begin = i * r.row_size;
		//			const auto end = begin + r.row_size;
		//			double sum = 0;
		//			for (auto j = begin; j < end; j++)
		//				sum += x[r.col_indices[j]];
		//			for (auto j = begin; j < end; j++)
		//				local_result[r.col_indices[j]] += static_cast<Vector::value_type>(sum);
		//		}

		//		#pragma omp critical
		//		{
		//			result += local_result;
		//		}
		//	}
		//	return result;
		//}
	};
}

inline auto transposed(const Matrix& m) { return TransposedMatrix(m); }

inline auto operator*(const Matrix& l, const Matrix& r) { return TwoMatrix(l, r); }
inline auto operator*(const Matrix& A, const Vector& x) { return A.Ax(x); }
