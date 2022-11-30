#include "Player.h"

Field FixedDepthPlayer::ChooseMove(const Position& pos)
{
	Moves moves = PossibleMoves(pos);
	if (moves.empty())
		return Field::invalid;

	if (intensity == 0) // random
	{
		std::size_t rnd = std::uniform_int_distribution<std::size_t>(0, moves.size() - 1)(rnd_engine);
		return moves[rnd];
	}

	return alg.Eval(pos, intensity).move;
}