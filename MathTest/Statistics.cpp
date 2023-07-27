#include "pch.h"
#include "Matrix.h"
#include "Vector.h"
#include <vector>

TEST(Statistics, Average)
{
	auto square = [](auto x) { return x * x; };
	EXPECT_EQ(Average(CreateVector(1)), 1);
	EXPECT_EQ(Average(CreateVector(1, 1)), 1);
	EXPECT_EQ(Average(CreateVector(1, 2)), 1.5);
	EXPECT_EQ(Average(CreateVector(1, 2), square), 2.5);
}

TEST(Statistics, Variance)
{
	auto square = [](auto x) { return x * x; };
	EXPECT_EQ(Variance(CreateVector(1)), 0);
	EXPECT_EQ(Variance(CreateVector(1, 1)), 0);
	EXPECT_EQ(Variance(CreateVector(1, 2)), 0.25);
	EXPECT_EQ(Variance(CreateVector(1, 2), square), 2.25);
}

TEST(Statistics, StandardDeviation)
{
	auto square = [](auto x) { return x * x; };
	EXPECT_EQ(StandardDeviation(CreateVector(1)), 0);
	EXPECT_EQ(StandardDeviation(CreateVector(1, 1)), 0);
	EXPECT_EQ(StandardDeviation(CreateVector(1, 2)), 0.5);
	EXPECT_EQ(StandardDeviation(CreateVector(1, 2), square), 1.5);
}

TEST(Statistics, Covariance_X_Y)
{
	std::vector<int> X{1, 2, 3, 4, 5};
	std::vector<int> Y{2, 4, 6, 8, 10};
	float result = Covariance(X, Y);
	EXPECT_DOUBLE_EQ(result, 4);
}

TEST(Statistics, Covariance_matrix)
{
	Matrix input = CreateMatrix2x2(-1, +1, +1, -1);
	auto cov = Covariance(input);
	EXPECT_EQ(cov, CreateMatrix2x2(+1, -1, -1, +1));
}

TEST(Statistics, Correlation)
{
	Matrix X = CreateMatrix3x3(1, 2, 3, 2, 4, 7, 1, 2, 0);
	Matrix result = Correlation(X);
	EXPECT_NEAR(result(0, 0), 1.0, 0.0001);
	EXPECT_NEAR(result(0, 1), 0.9934, 0.0001);
	EXPECT_NEAR(result(0, 2), -0.5, 0.0001);
	EXPECT_NEAR(result(1, 0), 0.9934, 0.0001);
	EXPECT_NEAR(result(1, 1), 1.0, 0.0001);
	EXPECT_NEAR(result(1, 2), -0.5960, 0.0001);
	EXPECT_NEAR(result(2, 0), -0.5, 0.0001);
	EXPECT_NEAR(result(2, 1), -0.5960, 0.0001);
	EXPECT_NEAR(result(2, 2), 1.0, 0.0001);
}

TEST(Statistics, AIC)
{
	std::vector<float> errors{1, 2, 3, 4, 5};
	float result = AIC(errors, 2);
	EXPECT_FLOAT_EQ(result, 7.4657359027997261);
}

TEST(Statistics, BIC)
{
	std::vector<float> errors{1, 2, 3, 4, 5};
	float result = BIC(errors, 2);
	EXPECT_FLOAT_EQ(result, 6.6846117276679271);
}
