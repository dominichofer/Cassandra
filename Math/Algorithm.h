#pragma once
#include <cassert>
#include <cmath>
#include <tuple>
#include "Vector.h"

inline Vector sqrt(Vector x)
{
	const int64_t size = x.size();
	#pragma omp parallel for schedule(dynamic, 64)
	for (int64_t i = 0; i < size; i++)
		x[i] = sqrt(x[i]);
	return x;
}

inline Vector::value_type dot(const Vector& a, const Vector& b)
{
	assert(a.size() == b.size());

	const int64_t size = a.size();
	Vector::value_type sum = 0;
	#pragma omp parallel for reduction(+ : sum)
	for (int64_t i = 0; i < size; i++)
		sum += a[i] * b[i];
	return sum;
}

inline Vector::value_type norm(const Vector& v)
{
	return std::sqrt(dot(v, v));
}


inline std::tuple<double, Vector> decompose(const Vector& x)
{
	const auto n = norm(x);
	return std::tuple(n, x / n);
}