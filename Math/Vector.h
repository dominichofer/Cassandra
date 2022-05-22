#pragma once
#include <cmath>
#include <numeric>
#include <valarray>

//template <typename T>
//double inv(const std::valarray<T>& l, T infinity)
//{
//	return std::inner_product(std::begin(l), std::end(l), std::begin(r), 0.0);
//}

template <typename T>
double dot(const std::valarray<T>& l, const std::valarray<T>& r)
{
	return std::inner_product(std::begin(l), std::end(l), std::begin(r), 0.0);
}

template <typename T>
double norm(const std::valarray<T>& x)
{
	using std::sqrt;
	return sqrt(dot(x, x));
}

template <typename T>
double L1_norm(const std::valarray<T>& x)
{
	using std::abs;
	return std::accumulate(std::begin(x), std::end(x), 0.0, [](const T& t) { return abs(t); });
}

template <typename T>
double L2_norm(const std::valarray<T>& x)
{
	return norm(x);
}
