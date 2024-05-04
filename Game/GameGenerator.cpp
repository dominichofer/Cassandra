#include "GameGenerator.h"

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

		if (move == Field::PS)
			pass_count++;
		else
		{
			game.Play(move);
			pass_count = 0;
		}

		first_to_play = not first_to_play;
	}
	return game;
}

std::vector<Game> PlayedGames(Player& first, Player& second, const std::vector<Position>& starts)
{
	std::vector<Game> ret(starts.size());
	#pragma omp parallel for
	for (int64_t i = 0; i < static_cast<int64_t>(starts.size()); i++)
		ret[i] = PlayedGame(first, second, starts[i]);
	return ret;
}
