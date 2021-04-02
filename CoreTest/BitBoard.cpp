#include "pch.h"

namespace Flip
{
	TEST(FlipDiagonal, bitwise)
	{
		for (int i = 0; i < 8; i++)
			for (int j = 0; j < 8; j++)
				ASSERT_EQ(FlipDiagonal(BitBoard::Bit(i, j)), BitBoard::Bit(j, i));
	}

	TEST(FlipCodiagonal, bitwise)
	{
		for (int i = 0; i < 8; i++)
			for (int j = 0; j < 8; j++)
				ASSERT_EQ(FlipCodiagonal(BitBoard::Bit(i, j)), BitBoard::Bit(7 - j, 7 - i));
	}

	TEST(FlipVertical, bitwise)
	{
		for (int i = 0; i < 8; i++)
			for (int j = 0; j < 8; j++)
				ASSERT_EQ(FlipVertical(BitBoard::Bit(i, j)), BitBoard::Bit(i, 7 - j));
	}

	TEST(FlipHorizontal, bitwise)
	{
		for (int i = 0; i < 8; i++)
			for (int j = 0; j < 8; j++)
				ASSERT_EQ(FlipHorizontal(BitBoard::Bit(i, j)), BitBoard::Bit(7 - i, j));
	}
}