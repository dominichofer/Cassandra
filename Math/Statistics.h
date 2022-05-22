#pragma once
#include <cmath>
#include <cstdint>
#include <functional>
#include <ranges>
#include "range/v3/all.hpp"

template <typename I, typename S, typename P = std::identity>
double Average(I first, S last, P proj = {})
{
	double E_of_X = 0;
	for (int64_t n = 1; first != last; ++first, ++n)
		E_of_X += (proj(*first) - E_of_X) / static_cast<double>(n);
	return E_of_X;
}

template <std::ranges::range Range, typename P = std::identity>
double Average(Range&& r, P proj = {})
{
	return Average(std::begin(r), std::end(r), std::move(proj));
}

// population variance
template <typename I, typename S, typename P = std::identity>
double Variance(I first, S last, P proj = {})
{
	// TODO: Use https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

	double E_of_X = 0;
	double E_of_X_sq = 0;
	for (int64_t n = 1; first != last; ++first, ++n)
	{
		auto x = proj(*first);
		E_of_X += (x - E_of_X) / static_cast<double>(n);
		E_of_X_sq += (x * x - E_of_X_sq) / static_cast<double>(n);
	}
	return E_of_X_sq - E_of_X * E_of_X;
}

// population variance
template <std::ranges::range Range, typename P = std::identity>
double Variance(Range&& r, P proj = {})
{
	return Variance(std::begin(r), std::end(r), std::move(proj));
}

// population standard deviation
template <typename I, typename S, typename P = std::identity>
double StandardDeviation(I first, S last, P proj = {})
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

	double meanx{ 0 }, meany{ 0 }, C{ 0 }, n{ 0 };
	for (auto&& [x, y] : ranges::views::zip(X, Y))
	{
		n++;
		double dx = x - meanx;
		meanx += dx / n;
		meany += (y - meany) / n;
		C += dx * (y - meany);
	}
	return C / n;
}

// population covariance
template <std::ranges::range Range1, std::ranges::range Range2, typename PX, typename PY>
double Covariance(Range1&& X, Range2&& Y, PX projX, PY projY)
{
	return Covariance(ranges::views::transform(X, projX), ranges::views::transform(Y, projY));
}

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
	double n = ranges::distance(X);
	return Covariance(X, Y) * (n / (n + 1));
}

// population covariance
template <typename T>
DenseMatrix<double> Covariance(const DenseMatrix<T>& X)
{
	// From https://en.wikipedia.org/wiki/Covariance_matrix

	DenseMatrix<double> cov(X.Rows(), X.Rows());
	for (std::size_t i = 0; i < X.Rows(); i++)
		for (std::size_t j = i; j < X.Rows(); j++)
			cov(i, j) = cov(j, i) = Covariance(X.Row(i), X.Row(j));
	return cov;
}

template <typename T>
DenseMatrix<double> Correlation(const DenseMatrix<T>& X)
{
	// From https://en.wikipedia.org/wiki/Correlation

	std::valarray<double> sd(X.Rows());
	for (std::size_t i = 0; i < sd.size(); i++)
		sd[i] = StandardDeviation(X.Row(i));

	DenseMatrix<double> corr(X.Rows(), X.Rows());
	for (std::size_t i = 0; i < X.Rows(); i++)
		for (std::size_t j = i; j < X.Rows(); j++)
			corr(i, j) = corr(j, i) = Covariance(X.Row(i), X.Row(j)) / (sd[i] * sd[j]);
	return corr;
}

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