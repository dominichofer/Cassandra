#pragma once
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <type_traits>

// population variance
template <typename I, typename S, typename P = std::identity>
double Variance(I first, S last, P proj = {})
{
	double E_of_X = 0;
	double E_of_X_sq = 0;
	for (int64_t n = 1; first != last; ++first, ++n)
	{
		const double x = proj(*first);
		E_of_X += (x - E_of_X) / n;
		E_of_X_sq += (x * x - E_of_X_sq) / n;
	}
	return E_of_X_sq - E_of_X * E_of_X;
}

// population variance
template <std::ranges::forward_range Range, typename P = std::identity>
double Variance(Range&& r, P proj = {})
{
	return Variance(r.begin(), r.end(), std::move(proj));
}

// population standard deviation
template <typename I, typename S, typename P = std::identity>
double StandardDeviation(I first, S last, P proj = {})
{
	return std::sqrt(Variance(first, last, std::move(proj)));
}

// population standard deviation
template <std::ranges::forward_range Range, typename P = std::identity>
double StandardDeviation(Range&& r, P proj = {})
{
	return StandardDeviation(r.begin(), r.end(), std::move(proj));
}

template <typename I, typename S, typename P = std::identity>
double Average(I first, S last, P proj = {})
{
	double E_of_X = 0;
	for (int64_t n = 1; first != last; ++first, ++n)
	{
		const double x = proj(*first);
		E_of_X += (x - E_of_X) / n;
	}
	return E_of_X;
}

template <std::ranges::forward_range Range, typename P = std::identity>
double Average(Range&& r, P proj = {})
{
	return Average(r.begin(), r.end(), std::move(proj));
}

// Bayesian Information Criterion
// for the gaussian special case https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case
// Good for selecting the best model if the true model is not in the set of candidates.
template <typename I, typename S, typename P = std::identity>
double AIC(I first_error, S last_error, std::size_t parameters, P proj = {})
{
	std::size_t n = std::distance(first_error, last_error);
	return n * log(Variance(first_error, last_error, std::move(proj))) + 2 * (parameters + 1);
}

// Bayesian Information Criterion
// for the gaussian special case https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case
// Good for selecting the best model if the true model is not in the set of candidates.
template <std::ranges::forward_range Range, typename P = std::identity>
double AIC(Range&& r, std::size_t parameters, P proj = {})
{
	return AIC(r.begin(), r.end(), parameters, std::move(proj));
}

// Bayesian Information Criterion
// for the gaussian special case https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case
// Good for selecting the true model if it's in the set of candidates.
template <typename I, typename S, typename P = std::identity>
double BIC(I first_error, S last_error, std::size_t parameters, P proj = {})
{
	std::size_t n = std::distance(first_error, last_error);
	return n * log(Variance(first_error, last_error, std::move(proj))) + parameters * log(n);
}

// Bayesian Information Criterion
// for the gaussian special case https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case
// Good for selecting the true model if it's in the set of candidates.
template <std::ranges::forward_range Range, typename P = std::identity>
double BIC(Range&& r, std::size_t parameters, P proj = {})
{
	return BIC(r.begin(), r.end(), parameters, std::move(proj));
}