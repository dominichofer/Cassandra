#include "pch.h"

TEST(Position, Equal)
{
    EXPECT_TRUE(Position(1, 2) == Position(1, 2));
    EXPECT_FALSE(Position(1, 2) == Position(1, 3));
    EXPECT_TRUE(Position(1, 2) != Position(1, 3));
    EXPECT_FALSE(Position(1, 2) != Position(1, 2));
}

TEST(Position, Population)
{
    EXPECT_EQ(Position(1, 2).Discs(), 3);
    EXPECT_EQ(~Position(1, 2).Empties(), 3);
    EXPECT_EQ(Position(1, 2).EmptyCount(), 62);
}

TEST(Position, SingleLine)
{
    EXPECT_EQ(
        SingleLine(Position::Start()),
        "---------------------------OX------XO--------------------------- X"
    );

    EXPECT_EQ(
        SingleLine(Position(0, 0xFFFFFFFFFFFFFFFFULL)),
        "OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO X"
    );
}

TEST(Position, MultiLine)
{
    EXPECT_EQ(
        MultiLine(Position::Start()),
        "  A B C D E F G H  \n"
        "1 - - - - - - - - 1\n"
        "2 - - - - - - - - 2\n"
        "3 - - - + - - - - 3\n"
        "4 - - + O X - - - 4\n"
        "5 - - - X O + - - 5\n"
        "6 - - - - + - - - 6\n"
        "7 - - - - - - - - 7\n"
        "8 - - - - - - - - 8\n"
        "  A B C D E F G H  "
    );
}

TEST(Position, PositionFromString)
{
    EXPECT_EQ(
        PositionFromString("---------------------------OX------XO--------------------------- X"),
        Position::Start()
    );

    EXPECT_EQ(
        PositionFromString("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX O"),
        Position(0, 0xFFFFFFFFFFFFFFFFULL)
    );

    EXPECT_THROW(PositionFromString("-------- X"), std::runtime_error);
}
