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

	auto operator*(const Vector& x) const { return Ax(x); }
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
	};
}

inline auto transposed(const Matrix& m) { return TransposedMatrix(m); }

namespace
{
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
	};
}

inline auto operator*(const Matrix& l, const Matrix& r) { return TwoMatrix(l, r); }
