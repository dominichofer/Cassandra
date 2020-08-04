#include "pch.h"

namespace Bit
{
	TEST(countr_zero, samples)
	{
		ASSERT_EQ(countr_zero(0), 64);
		ASSERT_EQ(countr_zero(1), 0);
		ASSERT_EQ(countr_zero(2), 1);
		ASSERT_EQ(countr_zero(3), 0);
		ASSERT_EQ(countr_zero(0x8000000000000000ULL), 63);
	}

	TEST(RemoveLSB, samples)
	{
		uint64_t a;
		a = 0; RemoveLSB(a); ASSERT_EQ(a, 0u);
		a = 1; RemoveLSB(a); ASSERT_EQ(a, 0u);
		a = 2; RemoveLSB(a); ASSERT_EQ(a, 0u);
		a = 3; RemoveLSB(a); ASSERT_EQ(a, 2u);
		a = 0x8000000001000000ULL; RemoveLSB(a); ASSERT_EQ(a, 0x8000000000000000ULL);
	}

	TEST(popcount, samples)
	{
		ASSERT_EQ(popcount(0), 0u);
		ASSERT_EQ(popcount(1), 1u);
		ASSERT_EQ(popcount(2), 1u);
		ASSERT_EQ(popcount(3), 2u);
		ASSERT_EQ(popcount(0xFFFFFFFFFFFFFFFFULL), 64u);
	}
}