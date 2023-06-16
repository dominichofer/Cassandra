#pragma once
#include "Field.h"
#include "Position.h"
#include <vector>

class Game
{
	Position start;
	std::vector<Field> moves;
public:
	Game(Position start = Position::Start(), std::vector<Field> moves = {}) noexcept;

	constexpr bool operator==(const Game& o) const noexcept { return (start == o.start) and (moves == o.moves); }
	constexpr bool operator!=(const Game& o) const noexcept { return !(*this == o); }

	Position StartPosition() const noexcept { return start; }
	const std::vector<Field>& Moves() const noexcept { return moves; }
	std::vector<Position> Positions() const;

	void Play(Field move) { moves.push_back(move); }
};

std::vector<Position> Positions(std::vector<Game>::const_iterator begin, std::vector<Game>::const_iterator end);
std::vector<Position> Positions(const std::vector<Game>&);