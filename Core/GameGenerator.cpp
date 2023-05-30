#include "GameGenerator.h"
#include "Player.h"

Game PlayedGame(Player& first, Player& second, Position start)
{
	Game game{ start };
	Position pos = start;

	bool first_to_play = true;
	int pass_count = 0;
	while (pass_count < 2)
	{
		Player& player = first_to_play ? first : second;

		Field move = player.ChooseMove(pos);
		pos = PlayOrPass(pos, move);

		if (move != Field::invalid)
			game.Play(move);

		first_to_play = not first_to_play;
		pass_count = (move == Field::invalid) ? pass_count + 1 : 0;
	}
	return game;
}

std::vector<Game> PlayedGamesFrom(Player& first, Player& second, const std::vector<Position>& starts)
{
	int64_t count = static_cast<int64_t>(starts.size());
	std::vector<Game> ret(count);
	#pragma omp parallel for
	for (int64_t i = 0; i < count; i++)
		ret[i] = PlayedGame(first, second, starts[i]);
	return ret;
}

Game SelfPlayedGame(Player& player, Position start)
{
	return PlayedGame(player, player, start);
}

std::vector<Game> SelfPlayedGamesFrom(Player& player, const std::vector<Position>& starts)
{
	return PlayedGamesFrom(player, player, starts);
}

Game RandomGame(Position start, unsigned int seed)
{
	RandomPlayer player{ seed };
	return SelfPlayedGame(player, start);
}

std::vector<Game> RandomGamesFrom(const std::vector<Position>& starts, unsigned int seed)
{
	int64_t count = static_cast<int64_t>(starts.size());
	std::vector<Game> ret(count, Game{});
	#pragma omp parallel for
	for (int64_t i = 0; i < count; i++)
		ret[i] = RandomGame(starts[i], seed + i);
	return ret;
}
