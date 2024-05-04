#include "Player.h"

Field RandomPlayer::ChooseMove(const Position& pos)
{
	Moves moves = PossibleMoves(pos);
	if (moves.empty())
		return Field::PS;

	auto dst = std::uniform_int_distribution<std::size_t>(0, moves.size() - 1);

	mtx.lock();
	std::size_t rnd = dst(rnd_engine);
	mtx.unlock();

	return static_cast<Field>(moves[rnd]);
}