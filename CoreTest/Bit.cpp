#include "pch.h"

namespace Bit
{
	TEST(Bit, countr_zero)
	{
		ASSERT_EQ(std::countr_zero(0ULL), 64);
		ASSERT_EQ(std::countr_zero(1ULL), 0);
		ASSERT_EQ(std::countr_zero(2ULL), 1);
		ASSERT_EQ(std::countr_zero(3ULL), 0);
		ASSERT_EQ(std::countr_zero(0x8000000000000000ULL), 63);
	}

	TEST(Bit, RemoveLSB)
	{
		uint64_t a;
		a = 0; RemoveLSB(a); ASSERT_EQ(a, 0ULL);
		a = 1; RemoveLSB(a); ASSERT_EQ(a, 0ULL);
		a = 2; RemoveLSB(a); ASSERT_EQ(a, 0ULL);
		a = 3; RemoveLSB(a); ASSERT_EQ(a, 2ULL);
		a = 0x8000000001000000ULL; RemoveLSB(a); ASSERT_EQ(a, 0x8000000000000000ULL);
	}

	TEST(Bit, popcount)
	{
		ASSERT_EQ(std::popcount(0ULL), 0);
		ASSERT_EQ(std::popcount(1ULL), 1);
		ASSERT_EQ(std::popcount(2ULL), 1);
		ASSERT_EQ(std::popcount(3ULL), 2);
		ASSERT_EQ(std::popcount(0xFFFFFFFFFFFFFFFFULL), 64);
	}
}