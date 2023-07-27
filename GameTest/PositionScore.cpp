#include "pch.h"

TEST(PosScore, IsPositionScore)
{
    EXPECT_TRUE(IsPositionScore("---------------------------OX------XO--------------------------- X % +12"));

    EXPECT_FALSE(IsPositionScore("-------- X % +12"));
}

TEST(PosScore, PosScoreFromString)
{
    EXPECT_EQ(
        PosScore(Position::Start(), +6),
        PosScoreFromString("---------------------------OX------XO--------------------------- X % +12")
    );

    EXPECT_THROW(PosScoreFromString("-------- X 1234"), std::runtime_error);
}
