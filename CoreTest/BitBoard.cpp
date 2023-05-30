#include "pch.h"

namespace BitBoardTests
{
	const BitBoard a =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - # #"_BitBoard;

	const BitBoard c =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"# - - - - - - -"
		"- # - - - - - -"_BitBoard;

	const BitBoard d =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - #"
		"- - - - - - # -"_BitBoard;

	const BitBoard h =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"# - - - - - - #"_BitBoard;

	const BitBoard v =
		"- - - - - - - #"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - #"_BitBoard;

	const BitBoard vh =
		"- # - - - - # -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- # - - - - # -"_BitBoard;

	const BitBoard dc =
		"- # - - - - - -"
		"# - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - #"
		"- - - - - - # -"_BitBoard;

	const BitBoard vhdc =
		"# - - - - - - #"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"# - - - - - - #"_BitBoard;

	TEST(IsSymmetric, Codiagonally)
	{
		ASSERT_EQ(a.IsCodiagonallySymmetric(), false);
		ASSERT_EQ(c.IsCodiagonallySymmetric(), true);
		ASSERT_EQ(d.IsCodiagonallySymmetric(), false);
		ASSERT_EQ(h.IsCodiagonallySymmetric(), false);
		ASSERT_EQ(v.IsCodiagonallySymmetric(), false);
		ASSERT_EQ(vh.IsCodiagonallySymmetric(), false);
		ASSERT_EQ(dc.IsCodiagonallySymmetric(), true);
		ASSERT_EQ(vhdc.IsCodiagonallySymmetric(), true);
	}

	TEST(IsSymmetric, Diagonally)
	{
		ASSERT_EQ(a.IsDiagonallySymmetric(), false);
		ASSERT_EQ(c.IsDiagonallySymmetric(), false);
		ASSERT_EQ(d.IsDiagonallySymmetric(), true);
		ASSERT_EQ(h.IsDiagonallySymmetric(), false);
		ASSERT_EQ(v.IsDiagonallySymmetric(), false);
		ASSERT_EQ(vh.IsDiagonallySymmetric(), false);
		ASSERT_EQ(dc.IsDiagonallySymmetric(), true);
		ASSERT_EQ(vhdc.IsDiagonallySymmetric(), true);
	}

	TEST(IsSymmetric, Horizontally)
	{
		ASSERT_EQ(a.IsHorizontallySymmetric(), false);
		ASSERT_EQ(c.IsHorizontallySymmetric(), false);
		ASSERT_EQ(d.IsHorizontallySymmetric(), false);
		ASSERT_EQ(h.IsHorizontallySymmetric(), true);
		ASSERT_EQ(v.IsHorizontallySymmetric(), false);
		ASSERT_EQ(vh.IsHorizontallySymmetric(), true);
		ASSERT_EQ(dc.IsHorizontallySymmetric(), false);
		ASSERT_EQ(vhdc.IsHorizontallySymmetric(), true);
	}

	TEST(IsSymmetric, Vertically)
	{
		ASSERT_EQ(a.IsVerticallySymmetric(), false);
		ASSERT_EQ(c.IsVerticallySymmetric(), false);
		ASSERT_EQ(d.IsVerticallySymmetric(), false);
		ASSERT_EQ(h.IsVerticallySymmetric(), false);
		ASSERT_EQ(v.IsVerticallySymmetric(), true);
		ASSERT_EQ(vh.IsVerticallySymmetric(), true);
		ASSERT_EQ(dc.IsVerticallySymmetric(), false);
		ASSERT_EQ(vhdc.IsVerticallySymmetric(), true);
	}

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
