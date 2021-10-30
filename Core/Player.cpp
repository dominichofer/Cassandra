#include "Player.h"

Position RandomPlayer::Play(const Position& pos)
{
	Moves possible_moves = PossibleMoves(pos);
	if (possible_moves.empty())
		return PlayPass(pos);

	int rnd = std::uniform_int_distribution<int>(0, possible_moves.size() - 1)(rnd_engine);
	auto it = possible_moves.begin();
	std::advance(it, rnd);

	return ::Play(pos, *it);
}
