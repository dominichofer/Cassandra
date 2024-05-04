#include "pch.h"
using namespace std::string_literals;
using namespace std::chrono_literals;

TEST(HH_MM_SS, 1ms)
{
	EXPECT_EQ(HH_MM_SS(1ms), "0.001");
}

TEST(HH_MM_SS, 1s)
{
	EXPECT_EQ(HH_MM_SS(1s), "1.000");
}

TEST(HH_MM_SS, 10s)
{
	EXPECT_EQ(HH_MM_SS(10s), "10.000");
}

TEST(HH_MM_SS, 1min)
{
	EXPECT_EQ(HH_MM_SS(1min), "1:00.000");
}

TEST(HH_MM_SS, 10min)
{
	EXPECT_EQ(HH_MM_SS(10min), "10:00.000");
}

TEST(HH_MM_SS, 1h)
{
	EXPECT_EQ(HH_MM_SS(1h), "1:00:00.000");
}

TEST(ShortTimeString, positive)
{
	EXPECT_EQ("1.000 s"s, ShortTimeString(1s));
	EXPECT_EQ("100.0 ms"s, ShortTimeString(100ms));
	EXPECT_EQ("10.00 ms"s, ShortTimeString(10ms));
	EXPECT_EQ("1.000 ms"s, ShortTimeString(1ms));
	EXPECT_EQ("100.0 us"s, ShortTimeString(100us));
	EXPECT_EQ("10.00 us"s, ShortTimeString(10us));
	EXPECT_EQ("1.000 us"s, ShortTimeString(1us));
	EXPECT_EQ("100.0 ns"s, ShortTimeString(100ns));
	EXPECT_EQ("10.00 ns"s, ShortTimeString(10ns));
	EXPECT_EQ("1.000 ns"s, ShortTimeString(1ns));
}

TEST(ShortTimeString, negative)
{
	EXPECT_EQ("-1.000 s"s, ShortTimeString(-1s));
	EXPECT_EQ("-100.0 ms"s, ShortTimeString(-100ms));
	EXPECT_EQ("-10.00 ms"s, ShortTimeString(-10ms));
	EXPECT_EQ("-1.000 ms"s, ShortTimeString(-1ms));
	EXPECT_EQ("-100.0 us"s, ShortTimeString(-100us));
	EXPECT_EQ("-10.00 us"s, ShortTimeString(-10us));
	EXPECT_EQ("-1.000 us"s, ShortTimeString(-1us));
	EXPECT_EQ("-100.0 ns"s, ShortTimeString(-100ns));
	EXPECT_EQ("-10.00 ns"s, ShortTimeString(-10ns));
	EXPECT_EQ("-1.000 ns"s, ShortTimeString(-1ns));
}

TEST(ShortTimeString, zero)
{
	EXPECT_EQ("0.000 s"s, ShortTimeString(0s));
}


TEST(MetricPrefix, )
{
	EXPECT_THROW(MetricPrefix(-11), std::out_of_range);
	EXPECT_EQ("m", MetricPrefix(-1));
	EXPECT_EQ("", MetricPrefix(0));
	EXPECT_EQ("k", MetricPrefix(+1));
	EXPECT_THROW(MetricPrefix(+11), std::out_of_range);
}
