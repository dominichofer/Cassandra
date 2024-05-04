#include "pch.h"

TEST(ScoredGameTest, to_string)
{
    EXPECT_EQ(
        to_string(ScoredGame(Game(Position::Start(), { Field::A1, Field::B2 }), { +12, -2, 0 })),
        "---------------------------OX------XO--------------------------- X A1 B2 +24 -04 +00"
    );
}

TEST(ScoredGameTest, FromString)
{
    EXPECT_EQ(
        ScoredGame::FromString("---------------------------OX------XO--------------------------- X A1 B2 +24 -04 +00"),
        ScoredGame(Game(Position::Start(), { Field::A1, Field::B2 }), { +12, -2, 0 })
    );
}