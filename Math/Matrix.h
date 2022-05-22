#pragma once
#include <cstdint>
#include <stdexcept>
#include "Vector.h"

namespace
{
	// Proxy
	template <typename T>
	requires requires (const T& t) { t.Rows(); t.Cols(); }
	struct TransposedMatrix
	{
		const T& m;

		explicit TransposedMatrix(const T& m) noexcept : m(m) {}

		std::size_t Rows() const { return m.Cols(); }
		std::size_t Cols() const { return m.Rows(); }
	};
}

template <typename T, typename U>
std::valarray<U> operator*(const TransposedMatrix<T>& mat, const std::valarray<U>& x)
{
	return x * mat.m;
}

template <typename T, typename U>
std::valarray<U> operator*(const std::valarray<U>& x, const TransposedMatrix<T>& mat)
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
	requires requires (L l, R r) { l.Rows(); l.Cols(); r.Rows(); r.Cols(); }
	struct TwoMatrix
	{
		const L& l;
		const R& r;

		TwoMatrix(const L& l, const R& r) : l(l), r(r)
		{
			if (l.Cols() != r.Rows())
				throw std::runtime_error("Size mismatch.");
		}
		operator L() const { return fallback::operator*(l, r); }

		std::size_t Rows() const { return l.Rows(); }
		std::size_t Cols() const { return r.Cols(); }
	};
}

template <typename L, typename R, typename T>
std::valarray<T> operator*(const TwoMatrix<L, R>& mat, const std::valarray<T>& x)
{
	return mat.l * (mat.r * x);
}

template <typename L, typename R, typename T>
std::valarray<T> operator*(const std::valarray<T>& x, const TwoMatrix<L, R>& mat)
{
	return (x * mat.l) * mat.r;
}

template <typename L, typename R>
requires requires (L l, R r) { l.Rows(); l.Cols(); r.Rows(); r.Cols(); }
auto operator*(const L& l, const R& r) { return TwoMatrix(l, r); }
