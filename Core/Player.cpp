#include "Player.h"

Position RandomPlayer::Play(const Position& pos)
{
	Moves possible_moves = PossibleMoves(pos);
	if (!possible_moves)
		return PlayPass(pos);

	auto rnd = std::uniform_int_distribution<int>(0, possible_moves.size() - 1)(rnd_engine);
	for (int i = 0; i < rnd; i++)
		possible_moves.RemoveFirst();

	return ::Play(pos, possible_moves.First());
}
