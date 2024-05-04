#include "pch.h"
#include <ranges>
#include <vector>

TEST(Game, to_string)
{
	EXPECT_EQ(
		to_string(Game(Position::Start(), { Field::A1, Field::B2 })),
		"---------------------------OX------XO--------------------------- X A1 B2"
	);
}

TEST(Game, FromString)
{
	EXPECT_EQ(
		Game::FromString("---------------------------OX------XO--------------------------- X A1 B2"),
		Game(Position::Start(), { Field::A1, Field::B2 })
	);
}

TEST(Game, Positions)
{
	Game game{ Position::Start(), { Field::C4, Field::E3, Field::F2 } };

	std::vector<Position> positions;
	for (Position pos : Positions(game))
		positions.push_back(pos);

	// Reference positions
	Position pos_0 = Position::Start();
	Position pos_1 = Play(pos_0, Field::C4);
	Position pos_2 = Play(pos_1, Field::E3);
	Position pos_3 = Play(pos_2, Field::F2);
	
	EXPECT_EQ(positions.size(), 4);
	EXPECT_EQ(positions[0], pos_0);
	EXPECT_EQ(positions[1], pos_1);
	EXPECT_EQ(positions[2], pos_2);
	EXPECT_EQ(positions[3], pos_3);
}
