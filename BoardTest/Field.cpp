#include "pch.h"

TEST(Field, IsField)
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

TEST(Field, to_string)
{
    EXPECT_EQ(to_string(Field::A1), "A1");
    EXPECT_EQ(to_string(Field::H8), "H8");
    EXPECT_EQ(to_string(Field::PS), "PS");
}

TEST(Field, FieldFromString)
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
