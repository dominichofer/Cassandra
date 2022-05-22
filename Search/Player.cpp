#include "Player.h"

Position FixedDepthPlayer::Play(const Position& pos)
{
	Moves moves = PossibleMoves(pos);
	if (moves.empty())
		return PlayPass(pos);

	if (intensity == 0) // random
	{
		int rnd = std::uniform_int_distribution<int>(0, moves.size() - 1)(rnd_engine);
		return ::Play(pos, moves[rnd]);
	}

	Field move = alg.Eval_BestMove(pos, intensity).move;
	return ::Play(pos, move);
}

Field FixedDepthPlayer::ChooseMove(const Position& pos)
{
	Moves moves = PossibleMoves(pos);
	if (moves.empty())
		return Field::invalid;

	if (intensity == 0) // random
	{
		int rnd = std::uniform_int_distribution<int>(0, moves.size() - 1)(rnd_engine);
		return moves[rnd];
	}

	return alg.Eval_BestMove(pos, intensity).move;
}