#include "pch.h"

TEST(PlayedGame, works_with_random_player)
{
	auto player_1 = RandomPlayer{};
	auto player_2 = RandomPlayer{};
	Game game = PlayedGame(player_1, player_2, Position::Start());

	ASSERT_GT(game.Positions().size(), 1);
}

TEST(PlayedGamesFrom, works_with_random_player)
{
	auto player_1 = RandomPlayer{};
	auto player_2 = RandomPlayer{};
	std::vector<Position> starts = { Position::Start(), Position::Start() };
	std::vector<Game> games = PlayedGamesFrom(player_1, player_2, starts);

	ASSERT_EQ(games.size(), 2);
}
