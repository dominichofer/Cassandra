#include "pch.h"
#include "DenseMatrix.h"

TEST(Statistics, Average)
{
	auto square = [](auto x) { return x * x; };
	EXPECT_EQ(Average(std::vector{ 1 }), 1);
	EXPECT_EQ(Average(std::vector{ 1,1 }), 1);
	EXPECT_EQ(Average(std::vector{ 1,2 }), 1.5);
	EXPECT_EQ(Average(std::vector{ 1,2 }, square), 2.5);
}

TEST(Statistics, Variance)
{
	auto square = [](auto x) { return x * x; };
	EXPECT_EQ(Variance(std::vector{ 1 }), 0);
	EXPECT_EQ(Variance(std::vector{ 1,1 }), 0);
	EXPECT_EQ(Variance(std::vector{ 1,2 }), 0.25);
	EXPECT_EQ(Variance(std::vector{ 1,2 }, square), 2.25);
}

TEST(Statistics, StandardDeviation)
{
	auto square = [](auto x) { return x * x; };
	EXPECT_EQ(StandardDeviation(std::vector{ 1 }), 0);
	EXPECT_EQ(StandardDeviation(std::vector{ 1,1 }), 0);
	EXPECT_EQ(StandardDeviation(std::vector{ 1,2 }), 0.5);
	EXPECT_EQ(StandardDeviation(std::vector{ 1,2 }, square), 1.5);
}

TEST(Statistics, Covariance)
{
	DenseMatrix<double> input = Matrix<double>(-1, +1, +1, -1);
	auto cov = Covariance(input);
	EXPECT_EQ(cov, Matrix<double>(+1, -1, -1, +1));
}

// TODO: Test AIC!
// TODO: Test BIC!
