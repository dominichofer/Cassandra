#include "Game.h"
#include "Core/Core.h"

using namespace GameGen;


Game Random::operator()() noexcept
{
	Position pos = Position::Start();
	Game game;

	while (true)
	{
		Moves possible_moves = PossibleMoves(pos);
		if (possible_moves.empty())
		{
			pos = PlayPass(pos);
			possible_moves = PossibleMoves(pos);
			if (possible_moves.empty())
				return game;
		}
		int rnd = std::uniform_int_distribution<int>(0, possible_moves.size() - 1)(rnd_engine);
		Field move = possible_moves[rnd];
		game.Add(move);
		pos = Play(pos, move);
	}
}

Played::Played(Player& first, Player& second, std::vector<Position> start)
	: first(first), second(second), start(std::move(start))
{
	start_pick = std::uniform_int_distribution<std::size_t>(0, this->start.size() - 1);
}

Game Played::operator()() noexcept
{
	Position pos = start[start_pick(rnd_engine)];
	Game game(pos);
	
	while (true)
	{
		Field move_first = first.ChooseMove(pos);
		if (move_first == Field::invalid)
			pos = PlayPass(pos);
		else
		{
			game.Add(move_first);
			pos = Play(pos, move_first);
		}

		Field second_move = second.ChooseMove(pos);
		if (second_move == Field::invalid)
		{
			if (move_first == Field::invalid)
				return game;
			pos = PlayPass(pos);
		}
		else
		{
			game.Add(second_move);
			pos = Play(pos, second_move);
		}
	}
}