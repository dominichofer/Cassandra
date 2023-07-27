#include "GameClass.h"
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

bool IsGame(std::string_view str)
{
	std::regex pattern("[XO-]{64} [XO]( [A-H][1-8])*");
	return std::regex_match(str.begin(), str.end(), pattern);
}

std::string to_string(const Game& game)
{
	std::string str = to_string(game.StartPosition());
	for (Field move : game.Moves())
		str += ' ' + to_string(move);
	return str;
}

Game GameFromString(std::string_view str)
{
	Position pos = PositionFromString(str);

	std::vector<Field> moves;
	for (int i = 67; i < str.length(); i += 3)
		moves.push_back(FieldFromString(str.substr(i, 2)));

	return { pos, std::move(moves) };
}