#include "GameClass.h"
#include <regex>

Position ImplicitPass(const Position& pos) noexcept
{
	if (not PossibleMoves(pos))
	{
		Position passed = PlayPass(pos);
		if (PossibleMoves(passed))
			return passed;
	}
	return pos;
}

Game::Game(Position start, std::vector<Field> moves) noexcept
	: start(ImplicitPass(start))
	, moves(std::move(moves))
{}

Game Game::FromString(std::string_view str)
{
	Position pos = Position::FromString(str);

	std::vector<Field> moves;
	moves.reserve((str.length() - 67) / 3);
	for (int i = 67; i < str.length(); i += 3)
		moves.push_back(FieldFromString(str.substr(i, 2)));

	return { pos, std::move(moves) };
}

std::vector<Position> Game::Positions() const
{
	std::vector<Position> ret;
	ret.reserve(moves.size() + 1);
	ret.push_back(start);
	Position pos = start;
	for (Field move : moves)
	{
		pos = ImplicitPass(::Play(pos, move));
		ret.push_back(pos);
	}
	return ret;
}

std::string to_string(const Game& game)
{
	std::string pos = to_string(game.StartPosition());
	std::string moves = join(' ', game.Moves(), [](Field f) { return to_string(f); });
	return pos + " " + moves;
}
