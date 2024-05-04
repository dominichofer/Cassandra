#include "pch.h"
#include "gtest/gtest.h"

TEST(StabilityTest, StableEdges)
{
    EXPECT_EQ(StableEdges(Position{ 0, 0 }), 0);

    auto pos =
        "X O X - X X O X"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - X"
        "- - - - - - - O"
        "- - - - - - - -"
        "X - - - - - - -"
        "O O O O X O O O"_pos;
	EXPECT_EQ(StableEdges(pos), 0xC3000000000000FFULL);
}

TEST(StabilityTest, StableStonesOpponent)
{
    EXPECT_EQ(StableStonesOpponent(Position{ 0, 0 }), 0);

    auto pos =
        "X O X - X X O X"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - O"
        "- - - - - - - O"
        "- - - - - - O O"
        "X - - - - O O O"
        "O O O O X O O O"_pos;
    EXPECT_EQ(StableStonesOpponent(pos), 0x42000001010307F7ULL);
}
