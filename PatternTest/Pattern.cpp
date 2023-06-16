#include "pch.h"
#include "Pattern.h"

TEST(Metatest, PatternA)
{
	EXPECT_FALSE(IsCodiagonallySymmetric(pattern_a));
	EXPECT_FALSE(IsDiagonallySymmetric(pattern_a));
	EXPECT_FALSE(IsHorizontallySymmetric(pattern_a));
	EXPECT_FALSE(IsVerticallySymmetric(pattern_a));
}

TEST(Metatest, PatternC)
{
	EXPECT_TRUE(IsCodiagonallySymmetric(pattern_c));
	EXPECT_FALSE(IsDiagonallySymmetric(pattern_c));
	EXPECT_FALSE(IsHorizontallySymmetric(pattern_c));
	EXPECT_FALSE(IsVerticallySymmetric(pattern_c));
}

TEST(Metatest, PatternD)
{
	EXPECT_FALSE(IsCodiagonallySymmetric(pattern_d));
	EXPECT_TRUE(IsDiagonallySymmetric(pattern_d));
	EXPECT_FALSE(IsHorizontallySymmetric(pattern_d));
	EXPECT_FALSE(IsVerticallySymmetric(pattern_d));
}

TEST(Metatest, PatternH)
{
	EXPECT_FALSE(IsCodiagonallySymmetric(pattern_h));
	EXPECT_FALSE(IsDiagonallySymmetric(pattern_h));
	EXPECT_TRUE(IsHorizontallySymmetric(pattern_h));
	EXPECT_FALSE(IsVerticallySymmetric(pattern_h));
}

TEST(Metatest, PatternV)
{
	EXPECT_FALSE(IsCodiagonallySymmetric(pattern_v));
	EXPECT_FALSE(IsDiagonallySymmetric(pattern_v));
	EXPECT_FALSE(IsHorizontallySymmetric(pattern_v));
	EXPECT_TRUE(IsVerticallySymmetric(pattern_v));
}

TEST(Metatest, PatternVH)
{
	EXPECT_FALSE(IsCodiagonallySymmetric(pattern_vh));
	EXPECT_FALSE(IsDiagonallySymmetric(pattern_vh));
	EXPECT_TRUE(IsHorizontallySymmetric(pattern_vh));
	EXPECT_TRUE(IsVerticallySymmetric(pattern_vh));
}

TEST(Metatest, PatternDC)
{
	EXPECT_TRUE(IsCodiagonallySymmetric(pattern_dc));
	EXPECT_TRUE(IsDiagonallySymmetric(pattern_dc));
	EXPECT_FALSE(IsHorizontallySymmetric(pattern_dc));
	EXPECT_FALSE(IsVerticallySymmetric(pattern_dc));
}

TEST(Metatest, PatternVHDC)
{
	EXPECT_TRUE(IsCodiagonallySymmetric(pattern_vhdc));
	EXPECT_TRUE(IsDiagonallySymmetric(pattern_vhdc));
	EXPECT_TRUE(IsHorizontallySymmetric(pattern_vhdc));
	EXPECT_TRUE(IsVerticallySymmetric(pattern_vhdc));
}

TEST(Metatest, SymmetricVariants)
{
	EXPECT_EQ(SymmetricVariants(pattern_h).size(), std::size_t(4));
	EXPECT_EQ(SymmetricVariants(pattern_d).size(), std::size_t(4));
	EXPECT_EQ(SymmetricVariants(pattern_a).size(), std::size_t(8));
}