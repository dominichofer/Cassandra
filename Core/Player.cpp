#include "Player.h"

Position RandomPlayer::Play(const Position& pos)
{
	Moves possible_moves = PossibleMoves(pos);
	if (possible_moves.empty())
		return PlayPass(pos);

	std::size_t rnd = std::uniform_int_distribution<std::size_t>(0, possible_moves.size() - 1)(rnd_engine);
	return ::Play(pos, possible_moves[rnd]);
}

Field RandomPlayer::ChooseMove(const Position& pos)
{
	Moves possible_moves = PossibleMoves(pos);
	if (possible_moves.empty())
		return Field::invalid;

	std::size_t rnd = std::uniform_int_distribution<std::size_t>(0, possible_moves.size() - 1)(rnd_engine);
	return static_cast<Field>(rnd);
}