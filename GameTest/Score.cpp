#include "pch.h"

TEST(Score, IsScore)
{
    EXPECT_TRUE(IsScore("+12"));
    EXPECT_TRUE(IsScore("-34"));
    EXPECT_TRUE(IsScore("+00"));
    EXPECT_TRUE(IsScore("-00"));

    EXPECT_FALSE(IsScore("12"));
    EXPECT_FALSE(IsScore("+-12"));
    EXPECT_FALSE(IsScore("+123"));
    EXPECT_FALSE(IsScore("+0"));
    EXPECT_FALSE(IsScore("-0"));
}

TEST(Score, ScoreToString)
{
    EXPECT_EQ(ScoreToString(6), "+12");
    EXPECT_EQ(ScoreToString(-17), "-34");
    EXPECT_EQ(ScoreToString(1), "+02");
    EXPECT_EQ(ScoreToString(0), "+00");
}

TEST(Score, ScoreFromString)
{
    EXPECT_EQ(6, ScoreFromString("+12"));
    EXPECT_EQ(-17, ScoreFromString("-34"));
    EXPECT_EQ(1, ScoreFromString("+02"));
    EXPECT_EQ(0, ScoreFromString("+00"));
    EXPECT_EQ(0, ScoreFromString("-00"));

    EXPECT_THROW(ScoreFromString("+A"), std::runtime_error);
}
