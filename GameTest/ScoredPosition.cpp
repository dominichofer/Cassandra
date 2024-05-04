#include "pch.h"

TEST(ScoredPosition, FromString)
{
    EXPECT_EQ(
        ScoredPosition(Position::Start(), +6),
        ScoredPosition::FromString("---------------------------OX------XO--------------------------- X % +12")
    );

    EXPECT_THROW(ScoredPosition::FromString("-------- X 1234"), std::runtime_error);
}
