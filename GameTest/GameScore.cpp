#include "pch.h"

TEST(GameScoreTest, IsGameScore)
{
    EXPECT_TRUE(IsGame("---------------------------OX------XO--------------------------- X"));
    EXPECT_TRUE(IsGame("---------------------------OX------XO--------------------------- X A1"));

    EXPECT_FALSE(IsGame("-------- X"));
}

TEST(GameScoreTest, ToString)
{
    EXPECT_EQ(
        to_string(GameScore(Game(Position::Start(), { Field::A1, Field::B2 }), { +12, -2, 0 })),
        "---------------------------OX------XO--------------------------- X A1 B2 +24 -04 +00"
    );
}

TEST(GameScoreTest, GameScoreFromString)
{
    EXPECT_EQ(
        GameScoreFromString("---------------------------OX------XO--------------------------- X A1 B2 +24 -04 +00"),
        GameScore(Game(Position::Start(), { Field::A1, Field::B2 }), { +12, -2, 0 })
    );
}