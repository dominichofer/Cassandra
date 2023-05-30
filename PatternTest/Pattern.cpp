#include "pch.h"
#include "Pattern.h"

TEST(Metatest, PatternA)
{
	ASSERT_EQ(pattern_a.IsCodiagonallySymmetric(), false);
	ASSERT_EQ(pattern_a.IsDiagonallySymmetric(), false);
	ASSERT_EQ(pattern_a.IsHorizontallySymmetric(), false);
	ASSERT_EQ(pattern_a.IsVerticallySymmetric(), false);
}

TEST(Metatest, PatternC)
{
	ASSERT_EQ(pattern_c.IsCodiagonallySymmetric(), true);
	ASSERT_EQ(pattern_c.IsDiagonallySymmetric(), false);
	ASSERT_EQ(pattern_c.IsHorizontallySymmetric(), false);
	ASSERT_EQ(pattern_c.IsVerticallySymmetric(), false);
}

TEST(Metatest, PatternD)
{
	ASSERT_EQ(pattern_d.IsCodiagonallySymmetric(), false);
	ASSERT_EQ(pattern_d.IsDiagonallySymmetric(), true);
	ASSERT_EQ(pattern_d.IsHorizontallySymmetric(), false);
	ASSERT_EQ(pattern_d.IsVerticallySymmetric(), false);
}

TEST(Metatest, PatternH)
{
	ASSERT_EQ(pattern_h.IsCodiagonallySymmetric(), false);
	ASSERT_EQ(pattern_h.IsDiagonallySymmetric(), false);
	ASSERT_EQ(pattern_h.IsHorizontallySymmetric(), true);
	ASSERT_EQ(pattern_h.IsVerticallySymmetric(), false);
}

TEST(Metatest, PatternV)
{
	ASSERT_EQ(pattern_v.IsCodiagonallySymmetric(), false);
	ASSERT_EQ(pattern_v.IsDiagonallySymmetric(), false);
	ASSERT_EQ(pattern_v.IsHorizontallySymmetric(), false);
	ASSERT_EQ(pattern_v.IsVerticallySymmetric(), true);
}

TEST(Metatest, PatternVH)
{
	ASSERT_EQ(pattern_vh.IsCodiagonallySymmetric(), false);
	ASSERT_EQ(pattern_vh.IsDiagonallySymmetric(), false);
	ASSERT_EQ(pattern_vh.IsHorizontallySymmetric(), true);
	ASSERT_EQ(pattern_vh.IsVerticallySymmetric(), true);
}

TEST(Metatest, PatternDC)
{
	ASSERT_EQ(pattern_dc.IsCodiagonallySymmetric(), true);
	ASSERT_EQ(pattern_dc.IsDiagonallySymmetric(), true);
	ASSERT_EQ(pattern_dc.IsHorizontallySymmetric(), false);
	ASSERT_EQ(pattern_dc.IsVerticallySymmetric(), false);
}

TEST(Metatest, PatternVHDC)
{
	ASSERT_EQ(pattern_vhdc.IsCodiagonallySymmetric(), true);
	ASSERT_EQ(pattern_vhdc.IsDiagonallySymmetric(), true);
	ASSERT_EQ(pattern_vhdc.IsHorizontallySymmetric(), true);
	ASSERT_EQ(pattern_vhdc.IsVerticallySymmetric(), true);
}

TEST(Metatest, SymmetricVariants)
{
	ASSERT_EQ(SymmetricVariants(pattern_h).size(), std::size_t(4));
	ASSERT_EQ(SymmetricVariants(pattern_d).size(), std::size_t(4));
	ASSERT_EQ(SymmetricVariants(pattern_a).size(), std::size_t(8));
}