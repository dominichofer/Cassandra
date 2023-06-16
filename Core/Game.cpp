#include "Game.h"
#include <regex>

Position PassIfNeeded(const Position& pos) noexcept
{
	if (PossibleMoves(pos))
		return pos;

	Position passed = PlayPass(pos);
	if (PossibleMoves(passed))
		return passed;
	else
		return pos;
}

Game::Game(Position start, std::vector<Field> moves) noexcept
	: start(PassIfNeeded(start))
	, moves(std::move(moves))
{}

std::vector<Position> Game::Positions() const
{
	std::vector<Position> ret;
	Position pos = start;
	ret.push_back(pos);
	for (Field move : moves)
	{
		pos = PassIfNeeded(::Play(pos, move));
		ret.push_back(pos);
	}
	return ret;
}

std::vector<Position> Positions(std::vector<Game>::const_iterator begin, std::vector<Game>::const_iterator end)
{
	std::vector<Position> ret;
	for (auto it = begin; it != end; ++it)
		for (Position p : it->Positions())
			ret.push_back(p);
	return ret;
}
std::vector<Position> Positions(const std::vector<Game>& games)
{
	return Positions(games.begin(), games.end());
}