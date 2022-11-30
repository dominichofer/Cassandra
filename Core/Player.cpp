#include "Player.h"

Field RandomPlayer::ChooseMove(const Position& pos)
{
	Moves possible_moves = PossibleMoves(pos);
	if (possible_moves.empty())
		return Field::invalid;

	auto dst = std::uniform_int_distribution<std::size_t>(0, possible_moves.size() - 1);

	mtx.lock();
	std::size_t rnd = dst(rnd_engine);
	mtx.unlock();

	return static_cast<Field>(possible_moves[rnd]);
}