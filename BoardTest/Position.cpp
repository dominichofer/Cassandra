#include "pch.h"

TEST(Position, FromString)
{
    EXPECT_EQ(
        Position::FromString("---------------------------OX------XO--------------------------- X"),
        Position::Start()
    );

    EXPECT_EQ(
        Position::FromString("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX O"),
        Position(0, 0xFFFFFFFFFFFFFFFFULL)
    );

    EXPECT_THROW(Position::FromString("-------- X"), std::runtime_error);
}

TEST(Position, Equal)
{
    EXPECT_TRUE(Position(1, 2) == Position(1, 2));
    EXPECT_FALSE(Position(1, 2) == Position(1, 4));
    EXPECT_TRUE(Position(1, 2) != Position(1, 4));
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

TEST(Position, Play)
{
    Position reference =
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - O - - - -"
        "- - - O O - - -"
        "- - - O X - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"_pos;
    EXPECT_EQ(Play(Position::Start(), Field::D3), reference);
}

TEST(Position, PlayPass)
{
    Position reference =
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - X O - - -"
        "- - - O X - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"_pos;
    EXPECT_EQ(PlayPass(Position::Start()), reference);
}

TEST(Position, FlippedCodiagonal)
{
    Position pos =
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "O - - - - - - -"
        "X X X - - - - -"_pos;
    Position reference =
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "X - - - - - - -"
        "X - - - - - - -"
        "X O - - - - - -"_pos;
    EXPECT_EQ(FlippedCodiagonal(pos), reference);
}

TEST(Position, FlippedDiagonal)
{
    Position pos =
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "O - - - - - - -"
        "X X X - - - - -"_pos;
    Position reference =
        "- - - - - - O X"
        "- - - - - - - X"
        "- - - - - - - X"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"_pos;
    EXPECT_EQ(FlippedDiagonal(pos), reference);
}

TEST(Position, FlippedHorizontal)
{
    Position pos =
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "O - - - - - - -"
        "X X X - - - - -"_pos;
    Position reference =
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - O"
        "- - - - - X X X"_pos;
    EXPECT_EQ(FlippedHorizontal(pos), reference);
}

TEST(Position, FlippedVertical)
{
    Position pos =
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "O - - - - - - -"
        "X X X - - - - -"_pos;
    Position reference =
        "X X X - - - - -"
        "O - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"_pos;
    EXPECT_EQ(FlippedVertical(pos), reference);
}

TEST(Position, FlippedToUnique)
{
    Position pos1 =
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "- - - - - - - -"
        "O - - - - - - -"
        "X X X - - - - -"_pos;
    Position pos2 = FlippedCodiagonal(pos1);
    Position pos3 = FlippedDiagonal(pos1);
    Position pos4 = FlippedHorizontal(pos1);
    Position pos5 = FlippedVertical(pos1);
    Position pos6 = FlippedVertical(pos2);
    Position pos7 = FlippedVertical(pos3);
    Position pos8 = FlippedVertical(pos4);

    EXPECT_NE(FlippedToUnique(pos1), FlippedToUnique(Position::Start()));
    EXPECT_EQ(FlippedToUnique(pos1), FlippedToUnique(pos2));
    EXPECT_EQ(FlippedToUnique(pos1), FlippedToUnique(pos3));
    EXPECT_EQ(FlippedToUnique(pos1), FlippedToUnique(pos4));
    EXPECT_EQ(FlippedToUnique(pos1), FlippedToUnique(pos5));
    EXPECT_EQ(FlippedToUnique(pos1), FlippedToUnique(pos6));
    EXPECT_EQ(FlippedToUnique(pos1), FlippedToUnique(pos7));
    EXPECT_EQ(FlippedToUnique(pos1), FlippedToUnique(pos8));
}
