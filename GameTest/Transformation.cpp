#include "pch.h"

Position pos1{ 0, 0 };
Position pos2{ 0, 1 };

ScoredPosition scored_pos1{ pos1, +1 };
ScoredPosition scored_pos2{ pos2, -1 };

Game game1{ pos1, { Field::C4 } };
Game game2{ pos2, { Field::A4, Field::B3 } };

ScoredGame scored_game1{ game1, { +1, -1 } };
ScoredGame scored_game2{ game2, { +1, -1, +2 } };

std::vector<Position> pos = { pos1, pos2 };
std::vector<Game> games = { game1, game2 };
std::vector<ScoredGame> scored_games = { scored_game1, scored_game2 };
std::vector<ScoredPosition> scored_pos = { scored_pos1, scored_pos2 };

TEST(Transformation, Positions_of_Game)
{
	EXPECT_EQ(Positions(game1).size(), 2);
}

TEST(Transformation, Positions_of_ScoredGame)
{
	EXPECT_EQ(Positions(scored_game1).size(), 2);
}

TEST(Transformation, Positions_of_Game_range)
{
	EXPECT_EQ(Positions(games).size(), 5);
}

TEST(Transformation, Positions_of_ScoredGame_range)
{
	EXPECT_EQ(Positions(scored_games).size(), 5);
}

TEST(Transformation, Positions_of_ScoredPosition_range)
{
	EXPECT_EQ(Positions(scored_pos).size(), 2);
}

TEST(Transformation, Scores_of_ScoredGame)
{
	EXPECT_EQ(Scores(scored_game1), scored_game1.scores);
}

TEST(Transformation, Scores_of_ScoredPosition_range)
{
	std::vector<Score> reference = { scored_pos1.score, scored_pos2.score };
	EXPECT_EQ(Scores(scored_pos), reference);
}

TEST(Transformation, ScoredPositions_of_ScoredGame_range)
{
	EXPECT_EQ(ScoredPositions(scored_games).size(), 5);
}

TEST(Transformation, ScoredPositions_of_Position_Score_range)
{
	std::vector<Score> scores = { +1, -1 };

	auto result = ScoredPositions(pos, scores);

	auto reference = std::vector<ScoredPosition>{ scored_pos1, scored_pos2 };
	EXPECT_EQ(result, reference);
}
