#include "pch.h"

namespace Bits
{
	TEST(Bit, bitwise)
	{
		for (uint8_t i = 0; i < 64; i++)
			ASSERT_EQ(Bit(i), 1ui64 << i);
	}

	TEST(SetBit, bitwise)
	{
		for (uint8_t i = 0; i < 64; i++)
		{
			uint64_t a = 0;
			SetBit(a, i);
			ASSERT_EQ(a, 1ui64 << i);
		}
	}

	TEST(ResetBit, bitwise)
	{
		for (uint8_t i = 0; i < 64; i++)
		{
			uint64_t a = ~0ui64;
			ResetBit(a, i);
			ASSERT_EQ(a, ~(1ui64 << i));
		}
	}

	TEST(TestBit, sample)
	{
		const uint64_t a = 0x8000000000000001ui64;
		ASSERT_EQ(TestBit(a, 0), true);
		for (uint8_t i = 1; i < 63; i++)
			ASSERT_EQ(TestBit(a, i), false);
		ASSERT_EQ(TestBit(a, 63), true);
	}
	
	TEST(TestBits, sample)
	{
		ASSERT_TRUE(TestBits(0x8000000000000001ui64, 1ui64));
	}
}

namespace Flip
{
	uint64_t Bit(const uint64_t i, const uint64_t j)
	{
		return ::Bit(i * 8 + j);
	}

	TEST(FlipDiagonal, bitwise)
	{
		for (int i = 0; i < 8; i++)
			for (int j = 0; j < 8; j++)
				ASSERT_EQ(FlipDiagonal(Bit(i, j)), Bit(j, i));
	}

	TEST(FlipCodiagonal, bitwise)
	{
		for (int i = 0; i < 8; i++)
			for (int j = 0; j < 8; j++)
				ASSERT_EQ(FlipCodiagonal(Bit(i, j)), Bit(7 - j, 7 - i));
	}

	TEST(FlipVertical, bitwise)
	{
		for (int i = 0; i < 8; i++)
			for (int j = 0; j < 8; j++)
				ASSERT_EQ(FlipVertical(Bit(i, j)), Bit(7 - i, j));
	}

	TEST(FlipHorizontal, bitwise)
	{
		for (int i = 0; i < 8; i++)
			for (int j = 0; j < 8; j++)
				ASSERT_EQ(FlipHorizontal(Bit(i, j)), Bit(i, 7 - j));
	}
}

namespace TrailingBits
{
	TEST(BitScanLSB, samples)
	{
		ASSERT_EQ(BitScanLSB(1), 0u);
		ASSERT_EQ(BitScanLSB(2), 1u);
		ASSERT_EQ(BitScanLSB(3), 0u);
		ASSERT_EQ(BitScanLSB(0x8000000000000000ui64), 63u);
	}

	TEST(CountTrailingZeros, samples)
	{
		ASSERT_EQ(CountTrailingZeros(1), 0u);
		ASSERT_EQ(CountTrailingZeros(2), 1u);
		ASSERT_EQ(CountTrailingZeros(3), 0u);
		ASSERT_EQ(CountTrailingZeros(0x8000000000000000ui64), 63u);
	}

	TEST(GetLSB, samples)
	{
		ASSERT_EQ(GetLSB(0), 0u);
		ASSERT_EQ(GetLSB(1), 1u);
		ASSERT_EQ(GetLSB(2), 2u);
		ASSERT_EQ(GetLSB(3), 1u);
		ASSERT_EQ(GetLSB(0x8000000000000000ui64), 0x8000000000000000ui64);
	}

	TEST(RemoveLSB, samples)
	{
		uint64_t a;
		a = 0; RemoveLSB(a); ASSERT_EQ(a, 0u);
		a = 1; RemoveLSB(a); ASSERT_EQ(a, 0u);
		a = 2; RemoveLSB(a); ASSERT_EQ(a, 0u);
		a = 3; RemoveLSB(a); ASSERT_EQ(a, 2u);
		a = 0x8000000001000000ui64; RemoveLSB(a); ASSERT_EQ(a, 0x8000000000000000ui64);
	}
}

namespace LeadingBits
{
	TEST(BitScanMSB, samples)
	{
		ASSERT_EQ(BitScanMSB(1), 0u);
		ASSERT_EQ(BitScanMSB(2), 1u);
		ASSERT_EQ(BitScanMSB(3), 1u);
		ASSERT_EQ(BitScanMSB(0x8000000000000000ui64), 63u);
	}

	TEST(CountLeadingZeros, samples)
	{
		ASSERT_EQ(CountLeadingZeros(1), 63u);
		ASSERT_EQ(CountLeadingZeros(2), 62u);
		ASSERT_EQ(CountLeadingZeros(3), 62u);
		ASSERT_EQ(CountLeadingZeros(0x8000000000000000ui64), 0u);
	}

	TEST(GetMSB, samples)
	{
		ASSERT_EQ(GetMSB(0), 0u);
		ASSERT_EQ(GetMSB(1), 1u);
		ASSERT_EQ(GetMSB(2), 2u);
		ASSERT_EQ(GetMSB(3), 2u);
		ASSERT_EQ(GetMSB(0x8000000000000000ui64), 0x8000000000000000ui64);
	}

	TEST(RemoveMSB, samples)
	{
		uint64_t a;
		a = 0; RemoveMSB(a); ASSERT_EQ(a, 0u);
		a = 1; RemoveMSB(a); ASSERT_EQ(a, 0u);
		a = 2; RemoveMSB(a); ASSERT_EQ(a, 0u);
		a = 3; RemoveMSB(a); ASSERT_EQ(a, 1u);
		a = 0x8000000001000000ui64; RemoveMSB(a); ASSERT_EQ(a, 0x0000000001000000ui64);
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
		ASSERT_EQ(PopCount(0xFFFFFFFFFFFFFFFFui64), 64u);
	}

	// TODO: Add test for BExtr
	// TODO: Add test for BZHI
	// TODO: Add test for PDep
	// TODO: Add test for PExt
	// TODO: Add test for BSwap
}