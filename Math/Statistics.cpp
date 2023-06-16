#include "Statistics.h"

Matrix Covariance(const Matrix& X)
{
	// From https://en.wikipedia.org/wiki/Covariance_matrix

	Matrix cov(X.Rows(), X.Rows());
	for (std::size_t i = 0; i < X.Rows(); i++)
		for (std::size_t j = i; j < X.Rows(); j++)
			cov(i, j) = cov(j, i) = Covariance(X.Row(i), X.Row(j));
	return cov;
}

Matrix Correlation(const Matrix& X)
{
	// From https://en.wikipedia.org/wiki/Correlation

	std::vector<double> sd(X.Rows());
	for (std::size_t i = 0; i < sd.size(); i++)
		sd[i] = StandardDeviation(X.Row(i));

	Matrix corr(X.Rows(), X.Rows());
	for (std::size_t i = 0; i < X.Rows(); i++)
		for (std::size_t j = i; j < X.Rows(); j++)
			corr(i, j) = corr(j, i) = Covariance(X.Row(i), X.Row(j)) / (sd[i] * sd[j]);
	return corr;
}