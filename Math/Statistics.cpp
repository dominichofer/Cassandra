#include "Statistics.h"

Matrix Covariance(const Matrix& X)
{
	// From https://en.wikipedia.org/wiki/Covariance_matrix

	int64_t size = X.Rows();
	Matrix cov(size, size);
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		for (std::size_t j = i; j < size; j++)
			cov(i, j) = cov(j, i) = Covariance(X.Row(i), X.Row(j));
	return cov;
}

Matrix Correlation(const Matrix& X)
{
	// From https://en.wikipedia.org/wiki/Correlation

	int64_t size = X.Rows();
	std::vector<float> sd(size);
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < sd.size(); i++)
		sd[i] = StandardDeviation(X.Row(i));

	Matrix corr(size, size);
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		for (int64_t j = i; j < size; j++)
			corr(i, j) = corr(j, i) = Covariance(X.Row(i), X.Row(j)) / (sd[i] * sd[j]);
	return corr;
}