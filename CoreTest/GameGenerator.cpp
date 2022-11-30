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

TEST(SelfPlayedGame, works_with_random_player)
{
	auto player = RandomPlayer{};
	Game game = SelfPlayedGame(player, Position::Start());

	ASSERT_GT(game.Positions().size(), 1);
}

TEST(SelfPlayedGamesFrom, works_with_random_player)
{
	auto player = RandomPlayer{};
	std::vector<Position> starts = { Position::Start(), Position::Start() };
	std::vector<Game> games = SelfPlayedGamesFrom(player, starts);

	ASSERT_EQ(games.size(), 2);
}

TEST(RandomGame, is_deterministic)
{
	unsigned int seed = 42; // arbitrary
	Game game_1 = RandomGame(Position::Start(), seed);
	Game game_2 = RandomGame(Position::Start(), seed);

	ASSERT_EQ(game_1, game_2);
}

TEST(RandomGamesFrom, is_deterministic)
{
	unsigned int seed = 42; // arbitrary
	int count = 3; // arbitrary
	std::vector<Position> starts(count, Position::Start());
	std::vector<Game> game_1 = RandomGamesFrom(starts, seed);
	std::vector<Game> game_2 = RandomGamesFrom(starts, seed);

	ASSERT_EQ(game_1, game_2);
}
