#pragma once
#include "Matrix.h"
#include <cmath>
#include <cstdint>
#include <functional>
#include <ranges>

struct WelfordResult
{
	double mean, variance, sample_variance;
};

template <typename T, typename P = std::identity>
WelfordResult Welford(T first, T last, P proj = {})
{
	// From https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

	double mean{ 0 }, M2{ 0 }, n{ 1 };
	for (; first != last; ++first, ++n)
	{
		double delta = proj(*first) - mean;
		mean += delta / n;
		double delta2 = proj(*first) - mean;
		M2 += delta * delta2;
	}
	return { mean, M2 / n, M2 / (n - 1) };
}

template <typename T, typename P = std::identity>
double Average(T first, T last, P proj = {})
{
	return Welford(first, last, std::move(proj)).mean;
}

template <std::ranges::range R, typename P = std::identity>
double Average(R&& r, P proj = {})
{
	return Average(std::begin(r), std::end(r), std::move(proj));
}

// population variance
template <typename T, typename P = std::identity>
double Variance(T first, T last, P proj = {})
{
	return Welford(first, last, std::move(proj)).sample_variance;
}

// population variance
template <std::ranges::range Range, typename P = std::identity>
double Variance(Range&& r, P proj = {})
{
	return Variance(std::begin(r), std::end(r), std::move(proj));
}

// population standard deviation
template <typename T, typename P = std::identity>
double StandardDeviation(T first, T last, P proj = {})
{
	using std::sqrt;
	return sqrt(Variance(first, last, std::move(proj)));
}

// population standard deviation
template <std::ranges::range Range, typename P = std::identity>
double StandardDeviation(Range&& r, P proj = {})
{
	return StandardDeviation(std::begin(r), std::end(r), std::move(proj));
}

// population covariance
template <std::ranges::range Range1, std::ranges::range Range2>
double Covariance(Range1&& X, Range2&& Y)
{
	// From https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online

	double mean_x{ 0 }, mean_y{ 0 }, C{ 0 }, n{ 0 };
	for (auto&& [x, y] : std::ranges::views::zip(X, Y))
	{
		n++;
		double dx = x - mean_x;
		mean_x += dx / n;
		mean_y += (y - mean_y) / n;
		C += dx * (y - mean_y);
	}
	return C / n;
}

// population covariance
template <std::ranges::range Range1, std::ranges::range Range2, typename PX, typename PY>
double Covariance(Range1&& X, Range2&& Y, PX projX, PY projY)
{
	return Covariance(std::ranges::views::transform(X, projX), std::ranges::views::transform(Y, projY));
}

// population covariance
Matrix Covariance(const Matrix&);

template <std::ranges::range Range1, std::ranges::range Range2>
double PopulationCovariance(Range1&& X, Range2&& Y)
{
	return Covariance(X, Y);
}

template <std::ranges::range Range1, std::ranges::range Range2, typename PX, typename PY>
double PopulationCovariance(Range1&& X, Range2&& Y, PX projX, PY projY)
{
	return Covariance(X, Y, projX, projY);
}

template <std::ranges::range Range1, std::ranges::range Range2>
double SampleCovariance(Range1&& X, Range2&& Y)
{
	double n = std::ranges::distance(X);
	return Covariance(X, Y) * (n / (n + 1));
}

Matrix Correlation(const Matrix&);

// Akaike Information Criterion
// for the gaussian special case.
// Good for selecting the best model if the true model is not in the set of candidates.
template <typename I, typename S, typename P = std::identity>
double AIC(I first_error, S last_error, std::size_t parameters, P proj = {})
{
	// From https://en.wikipedia.org/wiki/Akaike_information_criterion#Comparison_with_BIC
	using std::log;
	std::size_t n = std::distance(first_error, last_error);
	return n * log(Variance(first_error, last_error, std::move(proj))) + 2 * parameters;
}

// Akaike Information Criterion
// for the gaussian special case.
// Good for selecting the best model if the true model is not in the set of candidates.
template <std::ranges::range Range, typename P = std::identity>
double AIC(Range&& errors, std::size_t parameters, P proj = {})
{
	return AIC(std::begin(errors), std::end(errors), parameters, std::move(proj));
}

// Bayesian Information Criterion
// for the gaussian special case.
// Good for selecting the true model if it is in the set of candidates.
template <typename I, typename S, typename P = std::identity>
double BIC(I first_error, S last_error, std::size_t parameters, P proj = {})
{
	// From https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case
	using std::log;
	std::size_t n = std::distance(first_error, last_error);
	return n * log(Variance(first_error, last_error, std::move(proj))) + parameters * log(n);
}

// Bayesian Information Criterion
// for the gaussian special case.
// Good for selecting the true model if it is in the set of candidates.
template <std::ranges::range Range, typename P = std::identity>
double BIC(Range&& errors, std::size_t parameters, P proj = {})
{
	return BIC(std::begin(errors), std::end(errors), parameters, std::move(proj));
}