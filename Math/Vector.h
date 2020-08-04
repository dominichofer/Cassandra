#pragma once
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <limits>
#include <vector>
#include <omp.h>

class Vector
{
	using value_type = double;
private:
	std::vector<value_type> data;
public:
	Vector(std::size_t size = 0, value_type value = 0) : data(size, value) {}

	[[nodiscard]] auto operator==(const Vector& o) const noexcept { return data == o.data; }
	[[nodiscard]] auto operator!=(const Vector& o) const noexcept { return data != o.data; }

	std::size_t size() const noexcept { return data.size(); }
	auto push_back(value_type x) noexcept { return data.push_back(x); }
	void clear() noexcept { data.clear(); }

	value_type& operator[](std::size_t i) noexcept { return data[i]; }
	value_type& operator()(std::size_t i) noexcept { return data[i]; }
	const value_type& operator[](std::size_t i) const noexcept { return data[i]; }
	const value_type& operator()(std::size_t i) const noexcept { return data[i]; }

	Vector& operator+=(const Vector& x) 
	{
		assert(data.size() == x.size());
		const int64_t size = data.size();
		#pragma omp parallel for
		for (int64_t i = 0; i < size; i++)
			data[i] += x[i];
		return *this;
	}

	Vector& operator-=(const Vector& x)
	{
		assert(data.size() == x.size());
		const int64_t size = data.size();
		#pragma omp parallel for
		for (int64_t i = 0; i < size; i++)
			data[i] -= x[i];
		return *this;
	}

	Vector& operator*=(value_type m)
	{
		const int64_t size = data.size();
		#pragma omp parallel for
		for (int64_t i = 0; i < size; i++)
			data[i] *= m;
		return *this;
	}

	Vector& operator/=(value_type m) 
	{
		const int64_t size = data.size();
		#pragma omp parallel for
		for (int64_t i = 0; i < size; i++)
			data[i] /= m;
		return *this;
	}

	Vector elementwise_multiplication(Vector x) const
	{
		const int64_t size = data.size();
		#pragma omp parallel for
		for (int64_t i = 0; i < size; i++)
			x[i] *= data[i];
		return x;
	}

	Vector elementwise_division(Vector x) const
	{
		const int64_t size = data.size();
		#pragma omp parallel for
		for (int64_t i = 0; i < size; i++)
			x[i] /= data[i];
		return x;
	}
};

inline Vector operator+(      Vector  l, const Vector&  r) { return l += r; }
inline Vector operator+(const Vector& l,       Vector&& r) { return r += l; }

inline Vector operator-(      Vector  l, const Vector&  r) { return l -= r; }
inline Vector operator-(const Vector& l,       Vector&& r)
{
	assert(l.size() == r.size());
	const int64_t size = r.size();
	#pragma omp parallel for
	for (int64_t i = 0; i < size; i++)
		r[i] = l[i] - r[i];
	return r;
}

inline Vector operator*(Vector vec, double mul) { return vec *= mul; }
inline Vector operator*(double mul, Vector vec) { return vec *= mul; }

inline Vector operator/(Vector vec, double mul) { return vec /= mul; }

inline Vector inv(Vector x, double infinity = std::numeric_limits<double>::infinity())
{
	const int64_t size = x.size();
	#pragma omp parallel for
	for(int64_t i = 0; i < size; i++)
		x[i] = (x[i] == 0) ? infinity : 1.0 / x[i];
	return x;
}
