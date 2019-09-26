#include "Player.h"
#include "Machine/BitTwiddling.h"
#include "Machine.h"

Position RandomPlayer::Play(Position pos) noexcept(false)
{
	Moves moves = PossibleMoves(pos);
	if (moves.empty())
	{
		pos = PlayPass(pos);
		moves = PossibleMoves(pos);
		if (moves.empty())
			throw no_moves_exception();
		return pos;
	}

	auto rnd = std::uniform_int_distribution<std::size_t>(0, moves.size())(rnd_engine);
	for (std::size_t i = 0; i < rnd; i++)
		moves.pop_front();

	return ::Play(pos, moves.front());
}
