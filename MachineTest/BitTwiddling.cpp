#include "pch.h"

namespace TrailingBits
{
	TEST(BitScanLSB, samples)
	{
		ASSERT_EQ(BitScanLSB(1), 0u);
		ASSERT_EQ(BitScanLSB(2), 1u);
		ASSERT_EQ(BitScanLSB(3), 0u);
		ASSERT_EQ(BitScanLSB(0x8000000000000000ULL), 63u);
	}

	TEST(CountTrailingZeros, samples)
	{
		ASSERT_EQ(CountTrailingZeros(1), 0u);
		ASSERT_EQ(CountTrailingZeros(2), 1u);
		ASSERT_EQ(CountTrailingZeros(3), 0u);
		ASSERT_EQ(CountTrailingZeros(0x8000000000000000ULL), 63u);
	}

	TEST(GetLSB, samples)
	{
		ASSERT_EQ(GetLSB(0), 0u);
		ASSERT_EQ(GetLSB(1), 1u);
		ASSERT_EQ(GetLSB(2), 2u);
		ASSERT_EQ(GetLSB(3), 1u);
		ASSERT_EQ(GetLSB(0x8000000000000000ULL), 0x8000000000000000ULL);
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
}

namespace LeadingBits
{
	TEST(BitScanMSB, samples)
	{
		ASSERT_EQ(BitScanMSB(1), 0u);
		ASSERT_EQ(BitScanMSB(2), 1u);
		ASSERT_EQ(BitScanMSB(3), 1u);
		ASSERT_EQ(BitScanMSB(0x8000000000000000ULL), 63u);
	}

	TEST(CountLeadingZeros, samples)
	{
		ASSERT_EQ(CountLeadingZeros(1), 63u);
		ASSERT_EQ(CountLeadingZeros(2), 62u);
		ASSERT_EQ(CountLeadingZeros(3), 62u);
		ASSERT_EQ(CountLeadingZeros(0x8000000000000000ULL), 0u);
	}

	TEST(GetMSB, samples)
	{
		ASSERT_EQ(GetMSB(0), 0u);
		ASSERT_EQ(GetMSB(1), 1u);
		ASSERT_EQ(GetMSB(2), 2u);
		ASSERT_EQ(GetMSB(3), 2u);
		ASSERT_EQ(GetMSB(0x8000000000000000ULL), 0x8000000000000000ULL);
	}

	TEST(RemoveMSB, samples)
	{
		uint64_t a;
		a = 0; RemoveMSB(a); ASSERT_EQ(a, 0u);
		a = 1; RemoveMSB(a); ASSERT_EQ(a, 0u);
		a = 2; RemoveMSB(a); ASSERT_EQ(a, 0u);
		a = 3; RemoveMSB(a); ASSERT_EQ(a, 1u);
		a = 0x8000000001000000ULL; RemoveMSB(a); ASSERT_EQ(a, 0x0000000001000000ULL);
	}
}

namespace Intrinsics
{
	TEST(PopCount, samples)
	{
		ASSERT_EQ(PopCount(0), 0u);
		ASSERT_EQ(PopCount(1), 1u);
		ASSERT_EQ(PopCount(2), 1u);
		ASSERT_EQ(PopCount(3), 2u);
		ASSERT_EQ(PopCount(0xFFFFFFFFFFFFFFFFULL), 64u);
	}
}