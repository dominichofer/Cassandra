#include "Vector.h"
#include <cmath>
#include <numeric>
#include <omp.h>
#include <stdexcept>

void operator+=(Vector& l, const Vector& o)
{
	if (l.size() != o.size())
		throw std::runtime_error("Size mismatch");

	int64_t size = static_cast<int64_t>(l.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		l[i] += o[i];
}

void operator-=(Vector& l, const Vector& o)
{
	if (l.size() != o.size())
		throw std::runtime_error("Size mismatch");

	int64_t size = static_cast<int64_t>(l.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		l[i] -= o[i];
}

void operator*=(Vector& l, float factor)
{
	int64_t size = static_cast<int64_t>(l.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		l[i] *= factor;
}

void operator/=(Vector& l, float factor)
{
	int64_t size = static_cast<int64_t>(l.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		l[i] /= factor;
}

Vector elementwise_multiplication(Vector l, const Vector& o)
{
	if (l.size() != o.size())
		throw std::runtime_error("Size mismatch");

	int64_t size = static_cast<int64_t>(l.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		l[i] *= o[i];
	return l;
}

Vector elementwise_division(Vector l, const Vector& o)
{
	if (l.size() != o.size())
		throw std::runtime_error("Size mismatch");

	int64_t size = static_cast<int64_t>(l.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		l[i] /= o[i];
	return l;
}

Vector operator+(Vector l, const Vector& r)
{
	l += r;
	return l;
}

Vector operator+(const Vector& l, Vector&& r)
{
	r += l;
	return r;
}

Vector operator-(Vector l, const Vector& r)
{
	l -= r;
	return l;
}

Vector operator-(const Vector& l, Vector&& r)
{
	if (l.size() != r.size())
		throw std::runtime_error("Size mismatch");

	int64_t size = static_cast<int64_t>(l.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		r[i] = l[i] - r[i];
	return r;
}

Vector operator*(Vector v, float factor)
{
	v *= factor;
	return v;
}

Vector operator*(float factor, Vector v)
{
	v *= factor;
	return v;
}

Vector operator/(Vector v, float factor)
{
	v /= factor;
	return v;
}

Vector inv(Vector v)
{
	int64_t size = static_cast<int64_t>(v.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		v[i] = 1.0f / v[i];
	return v;
}

float dot(const Vector& l, const Vector& r)
{
	if (l.size() != r.size())
		throw std::runtime_error("Size mismatch");

	int64_t size = static_cast<int64_t>(l.size());
	float result = 0.0f;
	#pragma omp parallel for schedule(static) reduction(+:result)
	for (int64_t i = 0; i < size; i++)
		result += l[i] * r[i];
	return result;
}

float norm(const Vector& x)
{
	return std::sqrt(dot(x, x));
}

float L1_norm(const Vector& x)
{
	int64_t size = static_cast<int64_t>(x.size());
	float result = 0.0f;
	#pragma omp parallel for schedule(static) reduction(+:result)
	for (int64_t i = 0; i < size; i++)
		result += std::abs(x[i]);
	return result;
}

float L2_norm(const Vector& x)
{
	return norm(x);
}

float sum(const Vector& x)
{
	int64_t size = static_cast<int64_t>(x.size());
	float result = 0.0f;
	#pragma omp parallel for schedule(static) reduction(+:result)
	for (int64_t i = 0; i < size; i++)
		result += x[i];
	return result;
}

std::string to_string(const Vector& v)
{
	using std::to_string;
	auto begin = v.begin();
	auto end = v.end();
	std::string str = "(" + to_string(*begin++);
	while (begin != end)
		str += "," + to_string(*begin++);
	return str + ")";
}
