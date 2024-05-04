#include "pch.h"
#include <vector>

TEST(SampleTest, Zero)
{
	auto sample = Sample(0, std::vector<int>{ 1, 2, 3 }, /*seed*/ 13);
	EXPECT_EQ(sample, std::vector<int>{});
}

TEST(SampleTest, One)
{
	auto sample = Sample(1, std::vector<int>{ 1, 2, 3 }, /*seed*/ 13);
	EXPECT_EQ(sample.size(), 1);
}

TEST(SampleTest, All)
{
	auto sample = Sample(3, std::vector<int>{ 1, 2, 3 }, /*seed*/ 13);
	EXPECT_EQ(sample.size(), 3);
}

TEST(SampleTest, MoreThanPool)
{
	auto sample = Sample(4, std::vector<int>{ 1, 2, 3 }, /*seed*/ 13);
	EXPECT_EQ(sample.size(), 3);
}