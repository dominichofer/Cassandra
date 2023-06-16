#include "pch.h"
using namespace std::string_literals;
using namespace std::chrono_literals;
TEST(short_time_format, positive)
{
	EXPECT_EQ("1.000 s"s, short_time_format(1s));
	EXPECT_EQ("100.0 ms"s, short_time_format(100ms));
	EXPECT_EQ("10.00 ms"s, short_time_format(10ms));
	EXPECT_EQ("1.000 ms"s, short_time_format(1ms));
	EXPECT_EQ("100.0 us"s, short_time_format(100us));
	EXPECT_EQ("10.00 us"s, short_time_format(10us));
	EXPECT_EQ("1.000 us"s, short_time_format(1us));
	EXPECT_EQ("100.0 ns"s, short_time_format(100ns));
	EXPECT_EQ("10.00 ns"s, short_time_format(10ns));
	EXPECT_EQ("1.000 ns"s, short_time_format(1ns));
}

TEST(short_time_format, negative)
{
	EXPECT_EQ("-1.000 s"s, short_time_format(-1s));
	EXPECT_EQ("-100.0 ms"s, short_time_format(-100ms));
	EXPECT_EQ("-10.00 ms"s, short_time_format(-10ms));
	EXPECT_EQ("-1.000 ms"s, short_time_format(-1ms));
	EXPECT_EQ("-100.0 us"s, short_time_format(-100us));
	EXPECT_EQ("-10.00 us"s, short_time_format(-10us));
	EXPECT_EQ("-1.000 us"s, short_time_format(-1us));
	EXPECT_EQ("-100.0 ns"s, short_time_format(-100ns));
	EXPECT_EQ("-10.00 ns"s, short_time_format(-10ns));
	EXPECT_EQ("-1.000 ns"s, short_time_format(-1ns));
}

TEST(short_time_format, zero)
{
	EXPECT_EQ("0.000 s"s, short_time_format(0s));
}