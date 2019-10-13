#include "pch.h"

namespace Flip
{
	BitBoard Bit(const uint64_t i, const uint64_t j)
	{
		return BitBoard{ 1ui64 << (i * 8 + j) };
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