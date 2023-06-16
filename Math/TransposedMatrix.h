#pragma once
#include <cstdint>
#include <stdexcept>
#include "Vector.h"

namespace
{
	// Proxy
	template <typename T>
	struct TransposedMatrix
	{
		const T& m;

		explicit TransposedMatrix(const T& m) noexcept : m(m) {}

		std::size_t Rows() const noexcept { return m.Cols(); }
		std::size_t Cols() const noexcept { return m.Rows(); }
	};
}

template <typename T>
Vector operator*(const TransposedMatrix<T>& mat, const Vector& x)
{
	return x * mat.m;
}

template <typename T>
Vector operator*(const Vector& x, const TransposedMatrix<T>& mat)
{
	return mat.m * x;
}

template <typename T>
auto transposed(const T& m) { return TransposedMatrix(m); }

template <typename T>
auto transposed(const TransposedMatrix<T>& m) { return m.m; }


namespace
{
	// Proxy
	template <typename L, typename R>
	struct TwoMatrix
	{
		const L& l;
		const R& r;

		TwoMatrix(const L& l, const R& r) : l(l), r(r)
		{
			if (l.Cols() != r.Rows())
				throw std::runtime_error("Size mismatch.");
		}
		operator L() const { return ::operator*(l, r); }

		std::size_t Rows() const noexcept { return l.Rows(); }
		std::size_t Cols() const noexcept { return r.Cols(); }
	};
}

template <typename L, typename R>
Vector operator*(const TwoMatrix<L, R>& mat, const Vector& x)
{
	return mat.l * (mat.r * x);
}

template <typename L, typename R>
Vector operator*(const Vector& x, const TwoMatrix<L, R>& mat)
{
	return (x * mat.l) * mat.r;
}

template <typename L, typename R>
auto operator*(const L& l, const R& r) { return TwoMatrix(l, r); }
