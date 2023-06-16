#include "pch.h"

TEST(FieldTest, IsField)
{
    EXPECT_TRUE(IsField("A1"));
    EXPECT_TRUE(IsField("H8"));
    EXPECT_TRUE(IsField("PS"));

    EXPECT_FALSE(IsField("A0"));
    EXPECT_FALSE(IsField("A9"));
    EXPECT_FALSE(IsField("I8"));
    EXPECT_FALSE(IsField("B12"));
    EXPECT_FALSE(IsField("E"));
}

TEST(FieldTest, to_string)
{
    EXPECT_EQ(to_string(Field::A1), "A1");
    EXPECT_EQ(to_string(Field::H8), "H8");
    EXPECT_EQ(to_string(Field::PS), "PS");
}

TEST(FieldTest, FieldFromString)
{
    EXPECT_EQ(Field::A1, FieldFromString("A1"));
    EXPECT_EQ(Field::H8, FieldFromString("H8"));
    EXPECT_EQ(Field::PS, FieldFromString("PS"));

    EXPECT_THROW(FieldFromString("A0"), std::runtime_error);
    EXPECT_THROW(FieldFromString("A9"), std::runtime_error);
    EXPECT_THROW(FieldFromString("I8"), std::runtime_error);
    EXPECT_THROW(FieldFromString("B12"), std::runtime_error);
    EXPECT_THROW(FieldFromString("E"), std::runtime_error);
}

TEST(ScoreTest, IsScore)
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

TEST(ScoreTest, ScoreToString)
{
    EXPECT_EQ(ScoreToString(6), "+12");
    EXPECT_EQ(ScoreToString(-17), "-34");
    EXPECT_EQ(ScoreToString(1), "+02");
    EXPECT_EQ(ScoreToString(0), "+00");
}

TEST(ScoreTest, ScoreFromString)
{
    EXPECT_EQ(6, ScoreFromString("+12"));
    EXPECT_EQ(-17, ScoreFromString("-34"));
    EXPECT_EQ(1, ScoreFromString("+02"));
    EXPECT_EQ(0, ScoreFromString("+00"));
    EXPECT_EQ(0, ScoreFromString("-00"));

    EXPECT_THROW(ScoreFromString("+A"), std::runtime_error);
}

TEST(PositionTest, IsPosition)
{
    EXPECT_TRUE(IsPosition("XOXOXOXOXOXOXOXOXOXOXOXOXOXOXOXOXOXOXOXOXOXOXOXOXOXOXOXOXOXOXOXO X"));
    EXPECT_TRUE(IsPosition("---------------------------------------------------------------- O"));

    EXPECT_FALSE(IsPosition("----------------------------------------------------------------- X"));
    EXPECT_FALSE(IsPosition("--------------------------------------------------------------- X"));
}

TEST(PositionTest, SingleLine)
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

TEST(PositionTest, MultiLine)
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

TEST(PositionTest, PositionFromString)
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

TEST(PosScoreTest, IsPositionScore)
{
    EXPECT_TRUE(IsPositionScore("---------------------------OX------XO--------------------------- X % +12"));

    EXPECT_FALSE(IsPositionScore("-------- X % +12"));
}

TEST(PosScoreTest, PosScoreFromString)
{
    EXPECT_EQ(
        PosScore(Position::Start(), +6),
        PosScoreFromString("---------------------------OX------XO--------------------------- X % +12")
    );

    EXPECT_THROW(PosScoreFromString("-------- X 1234"), std::runtime_error);
}

TEST(GameTest, IsGame)
{
    EXPECT_TRUE(IsGame("---------------------------OX------XO--------------------------- X"));
    EXPECT_TRUE(IsGame("---------------------------OX------XO--------------------------- X A1"));

    EXPECT_FALSE(IsGame("-------- X"));
}

TEST(GameTest, ToString)
{
    EXPECT_EQ(
        to_string(Game(Position::Start(), { Field::A1, Field::B2 })),
        "---------------------------OX------XO--------------------------- X A1 B2"
    );
}

TEST(GameTest, GameFromString)
{
    EXPECT_EQ(
        GameFromString("---------------------------OX------XO--------------------------- X A1 B2"),
        Game(Position::Start(), { Field::A1, Field::B2 })
    );
}

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