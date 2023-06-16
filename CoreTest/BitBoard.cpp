#include "pch.h"
#include <cstdint>

namespace BitBoardTests
{
	TEST(Flipped, Codiagonal)
	{
		ASSERT_EQ(FlippedCodiagonal(0xFULL), 0x8080808000000000ULL);
	}

	TEST(Flipped, Diagonal)
	{
		ASSERT_EQ(FlippedDiagonal(0xFULL), 0x01010101ULL);
	}

	TEST(Flipped, Horizontal)
	{
		ASSERT_EQ(FlippedHorizontal(0xFULL), 0xF0ULL);
	}

	TEST(Flipped, Vertical)
	{
		ASSERT_EQ(FlippedVertical(0xFULL), 0x0F00000000000000ULL);
	}
}
