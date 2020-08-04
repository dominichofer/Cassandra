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

inline double dot(const Vector& a, const Vector& b)
{
	assert(a.size() == b.size());

	const int64_t size = a.size();
	double sum = 0;
	#pragma omp parallel for reduction(+ : sum)
	for (int64_t i = 0; i < size; i++)
		sum += a[i] * b[i];
	return sum;
}

inline double norm(const Vector& v)
{
	return std::sqrt(dot(v, v));
}

template <typename Vector>
double SampleStandardDeviation(const Vector& vec)
{
	double E_of_X = 0;
	double E_of_X_sq = 0;
	for (std::size_t i = 0; i < vec.size(); i++)
	{
		const double x = vec[i];
		const double N = static_cast<double>(i + 1);
		E_of_X += (x - E_of_X) / N;
		E_of_X_sq += (x * x - E_of_X_sq) / N;
	}
	return std::sqrt(E_of_X_sq - E_of_X * E_of_X);
}
