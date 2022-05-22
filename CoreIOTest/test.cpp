#include "pch.h"

TEST(Field, to_string_Parse)
{
	EXPECT_EQ(ParseField(to_string(Field::A1)), Field::A1);
	EXPECT_EQ(ParseField(to_string(Field::A2)), Field::A2);
	EXPECT_EQ(ParseField(to_string(Field::invalid)), Field::invalid);
	EXPECT_EQ(ParseField("arbitrary"), Field::invalid);
}

TEST(Position, to_string_Parse)
{
	Position pos;

	pos = Position::Start();
	EXPECT_EQ(ParsePosition_SingleLine(to_string(pos)), pos);

	pos = Position(1, 2); // arbitrary
	EXPECT_EQ(ParsePosition_SingleLine(to_string(pos)), pos);
}

TEST(Stream, serialize_deserialize_BitBoard)
{
	const BitBoard in = BitBoard::HorizontalLine(3);

	std::stringstream stream;
	Serialize(in, stream);
	auto out = Deserialize<decltype(in)>(stream);

	EXPECT_EQ(in, out);
}

TEST(Stream, serialize_deserialize_Position)
{
	const Position in = Position::Start();

	std::stringstream stream;
	Serialize(in, stream);
	auto out = Deserialize<decltype(in)>(stream);

	EXPECT_EQ(in, out);
}
