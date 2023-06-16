#include "Vector.h"
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <omp.h>

Vector& Vector::operator+=(const Vector& o)
{
	if (data.size() != o.size())
		throw std::runtime_error("Size mismatch");

	const int64_t size = static_cast<int64_t>(data.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		data[i] += o[i];
	return *this;
}

Vector& Vector::operator-=(const Vector& o)
{
	if (data.size() != o.size())
		throw std::runtime_error("Size mismatch");

	const int64_t size = static_cast<int64_t>(data.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		data[i] -= o[i];
	return *this;
}

Vector& Vector::operator*=(double factor)
{
	const int64_t size = static_cast<int64_t>(data.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		data[i] *= factor;
	return *this;
}

Vector& Vector::operator/=(double factor)
{
	const int64_t size = static_cast<int64_t>(data.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		data[i] /= factor;
	return *this;
}

Vector& Vector::elementwise_multiplication(const Vector& o)
{
	if (data.size() != o.size())
		throw std::runtime_error("Size mismatch");

	const int64_t size = static_cast<int64_t>(data.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		data[i] *= o[i];
	return *this;
}

Vector& Vector::elementwise_division(const Vector& o)
{
	if (data.size() != o.size())
		throw std::runtime_error("Size mismatch");

	const int64_t size = static_cast<int64_t>(data.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		data[i] /= o[i];
	return *this;
}

Vector operator+(Vector l, const Vector& r)
{
	return l += r;
}

Vector operator+(const Vector& l, Vector&& r)
{
	return r += l;
}

Vector operator-(Vector l, const Vector& r)
{
	return l -= r;
}

Vector operator-(const Vector& l, Vector&& r)
{
	if (l.size() != r.size())
		throw std::runtime_error("Size mismatch");

	const int64_t size = static_cast<int64_t>(l.size());
	#pragma omp parallel for
	for (int64_t i = 0; i < size; i++)
		r[i] = l[i] - r[i];
	return r;
}

Vector operator*(Vector v, double factor)
{
	return v *= factor;
}

Vector operator*(double factor, Vector v)
{
	return v *= factor;
}

Vector operator/(Vector v, double factor)
{
	return v /= factor;
}

Vector inv(Vector v)
{
	const int64_t size = static_cast<int64_t>(v.size());
	#pragma omp parallel for
	for (int64_t i = 0; i < size; i++)
		v[i] = 1.0 / v[i];
	return v;
}

double dot(const Vector& l, const Vector& r)
{
	if (l.size() != r.size())
		throw std::runtime_error("Size mismatch");

	return std::inner_product(std::begin(l), std::end(l), std::begin(r), 0.0);
}

double norm(const Vector& x)
{
	return std::sqrt(dot(x, x));
}

double L1_norm(const Vector& x)
{
	return std::transform_reduce(std::begin(x), std::end(x), 0, std::plus<>{}, static_cast<double (*)(double)>(std::abs));
}

double L2_norm(const Vector& x)
{
	return norm(x);
}

double sum(const Vector& x)
{
	return std::accumulate(std::begin(x), std::end(x), 0.0);
}
