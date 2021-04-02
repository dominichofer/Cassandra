#pragma once
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <type_traits>

template <class Iterator, class Function = std::identity>
double StandardDeviation(Iterator first, Iterator last, Function trafo = {})
{
	static_assert(std::is_convertible_v<std::iterator_traits<Iterator>::value_type, double>);

	double E_of_X = 0;
	double E_of_X_sq = 0;
	for (int64_t n = 1; first != last; ++first, ++n)
	{
		const double x = trafo(*first);
		E_of_X += (x - E_of_X) / n;
		E_of_X_sq += (x * x - E_of_X_sq) / n;
	}
	return std::sqrt(E_of_X_sq - E_of_X * E_of_X);
}

template <class Container, class Function = std::identity>
double StandardDeviation(const Container& c, Function trafo = {})
{
	return StandardDeviation(c.cbegin(), c.cend(), trafo);
}

template <class Iterator, class Function>
double Avg(Iterator first, Iterator last, Function trafo = std::identity)
{
	static_assert(std::is_convertible_v<std::iterator_traits<Iterator>::value_type, double>);

	double E_of_X = 0;
	for (int64_t n = 1; first != last; ++first, ++n)
	{
		const double x = trafo(*first);
		E_of_X += (x - E_of_X) / n;
	}
	return E_of_X;
}

template <class Container, class Function>
double Avg(const Container& c, Function trafo = std::identity)
{
	return Avg(c.cbegin(), c.cend(), trafo);
}