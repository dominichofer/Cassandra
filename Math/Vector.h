#pragma once
#include <cmath>
#include <numeric>
#include <vector>

template <typename T>
std::vector<T>& operator+=(std::vector<T>& l, const std::vector<T>& r)
{
	if (l.size() != r.size())
		throw std::runtime_error("Size mismatch");

	const int64_t size = static_cast<int64_t>(l.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		l[i] += r[i];
	return l;
}

template <typename T>
std::vector<T>& operator-=(std::vector<T>& l, const std::vector<T>& r)
{
	if (l.size() != r.size())
		throw std::runtime_error("Size mismatch");

	const int64_t size = static_cast<int64_t>(l.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		l[i] -= r[i];
	return l;
}

template <typename T>
std::vector<T>& operator*=(std::vector<T>& l, double factor)
{
	const int64_t size = static_cast<int64_t>(l.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		l[i] *= factor;
	return l;
}

template <typename T>
std::vector<T>& operator/=(std::vector<T>& l, double factor)
{
	const int64_t size = static_cast<int64_t>(l.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		l[i] /= factor;
	return l;
}

template <typename T>
std::vector<T> elementwise_multiplication(std::vector<T> l, const std::vector<T>& r)
{
	const int64_t size = static_cast<int64_t>(l.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		l[i] *= r[i];
	return l;
}

template <typename T>
std::vector<T> elementwise_division(std::vector<T> l, const std::vector<T>& r)
{
	const int64_t size = static_cast<int64_t>(l.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		l[i] /= r[i];
	return l;
}

template <typename T> std::vector<T> operator+(std::vector<T> l, const std::vector<T>& r) { return l += r; }
template <typename T> std::vector<T> operator+(const std::vector<T>& l, std::vector<T>&& r) { return r += l; }

template <typename T> std::vector<T> operator-(std::vector<T> l, const std::vector<T>& r) { return l -= r; }
template <typename T> std::vector<T> operator-(const std::vector<T>& l, std::vector<T>&& r)
{
	if (l.size() != r.size())
		throw std::runtime_error("Size mismatch");

	const int64_t size = static_cast<int64_t>(l.size());
	#pragma omp parallel for
	for (int64_t i = 0; i < size; i++)
		r[i] = l[i] - r[i];
	return r;
}

template <typename T> std::vector<T> operator*(std::vector<T> vec, double factor) { return vec *= factor; }
template <typename T> std::vector<T> operator*(double factor, std::vector<T> vec) { return vec *= factor; }

template <typename T> std::vector<T> operator/(std::vector<T> vec, double factor) { return vec /= factor; }

//template <typename T>
//double inv(const std::vector<T>& l, T infinity)
//{
//	return std::inner_product(std::begin(l), std::end(l), std::begin(r), 0.0);
//}

template <typename T>
double dot(const std::vector<T>& l, const std::vector<T>& r)
{
	return std::inner_product(std::begin(l), std::end(l), std::begin(r), 0.0);
}

template <typename T>
double norm(const std::vector<T>& x)
{
	using std::sqrt;
	return sqrt(dot(x, x));
}

template <typename T>
double sum(const std::vector<T>& x)
{
	return std::accumulate(std::begin(x), std::end(x), 0.0);
}

template <typename T>
double L1_norm(const std::vector<T>& x)
{
	using std::abs;
	return std::accumulate(std::begin(x), std::end(x), 0.0, [](const T& t) { return abs(t); });
}

template <typename T>
double L2_norm(const std::vector<T>& x)
{
	return norm(x);
}
