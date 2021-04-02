#pragma once
#include <cassert>
#include <cmath>
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
	double sum = 0;
	#pragma omp parallel for reduction(+ : sum)
	for (int64_t i = 0; i < size; i++)
		sum += a[i] * b[i];
	return static_cast<Vector::value_type>(sum);
}

inline Vector::value_type norm(const Vector& v)
{
	return static_cast<Vector::value_type>(std::sqrt(dot(v, v)));
}
