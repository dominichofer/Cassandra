#pragma once
#include <cstdint>
#include "Vector.h"

class IMatrix
{
public:
	virtual std::size_t Rows() const noexcept = 0;
	virtual std::size_t Cols() const noexcept = 0;
	virtual std::size_t size() const noexcept = 0;
	virtual std::size_t nnz() const noexcept = 0;

	virtual Vector operator*(const Vector& x) const = 0;
	virtual Vector ATAx(const Vector& x) const { return Vector(); }
};
