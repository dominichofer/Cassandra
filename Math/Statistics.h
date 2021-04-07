#pragma once
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <type_traits>

// population variance
template <class Iterator, class Function = std::identity>
double Variance(Iterator first, Iterator last, Function trafo = {})
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
	return E_of_X_sq - E_of_X * E_of_X;
}

// population variance
template <class Container, class Function = std::identity>
double Variance(const Container& c, Function trafo = {})
{
	return Variance(c.begin(), c.end(), trafo);
}

// population standard deviation
template <class Iterator, class Function = std::identity>
double StandardDeviation(Iterator first, Iterator last, Function trafo = {})
{
	return std::sqrt(Variance(first, last, trafo));
}

// population standard deviation
template <class Container, class Function = std::identity>
double StandardDeviation(const Container& c, Function trafo = {})
{
	return StandardDeviation(c.begin(), c.end(), trafo);
}

template <class Iterator, class Function = std::identity>
double Average(Iterator first, Iterator last, Function trafo = {})
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

template <class Container, class Function = std::identity>
double Average(const Container& c, Function trafo = {})
{
	return Average(c.begin(), c.end(), trafo);
}

// Bayesian Information Criterion
// for the gaussian special case https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case
// Good for selecting the best model if the true model is not in the set of candidates.
template <class Iterator, class Function = std::identity>
double AIC(Iterator first_error, Iterator last_error, std::size_t parameters, Function trafo = {})
{
	std::size_t n = std::distance(first_error, last_error);
	return n * log(Variance(first_error, last_error)) + 2 * (parameters + 1);
}

// Bayesian Information Criterion
// for the gaussian special case https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case
// Good for selecting the best model if the true model is not in the set of candidates.
template <class Container, class Function = std::identity>
double AIC(const Container& c, std::size_t parameters, Function trafo = {})
{
	std::size_t n = c.size();
	return n * log(Variance(c)) + 2 * (parameters + 1);
}

// Bayesian Information Criterion
// for the gaussian special case https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case
// Good for selecting the true model if it's in the set of candidates.
template <class Iterator, class Function = std::identity>
double BIC(Iterator first_error, Iterator last_error, std::size_t parameters, Function trafo = {})
{
	std::size_t n = std::distance(first_error, last_error);
	return n * log(Variance(first_error, last_error)) + parameters * log(n);
}

// Bayesian Information Criterion
// for the gaussian special case https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case
// Good for selecting the true model if it's in the set of candidates.
template <class Container, class Function = std::identity>
double BIC(const Container& c, std::size_t parameters, Function trafo = {})
{
	std::size_t n = c.size();
	return n * log(Variance(c)) + parameters * log(n);
}