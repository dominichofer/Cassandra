#pragma once
#include "Board/Board.h"
#include <string>
#include <string_view>
#include <vector>

class Game
{
	Position start;
	std::vector<Field> moves;
public:
	Game(Position start = Position::Start(), std::vector<Field> moves = {}) noexcept;

	bool operator==(const Game& o) const noexcept { return (start == o.start) and (moves == o.moves); }
	bool operator!=(const Game& o) const noexcept { return !(*this == o); }

	Position StartPosition() const noexcept { return start; }
	const std::vector<Field>& Moves() const noexcept { return moves; }
	std::vector<Position> Positions() const;

	void Play(Field move) { moves.push_back(move); }
};

bool IsGame(std::string_view);
std::string to_string(const Game&);
Game GameFromString(std::string_view);