#pragma once
#include "Board/Board.h"
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

class Game
{
	Position start;
	std::vector<Field> moves;
public:
	Game() noexcept = default;
	Game(Position start, std::vector<Field> moves = {}) noexcept;

	static Game FromString(std::string_view);

	bool operator==(const Game&) const noexcept = default;
	bool operator!=(const Game&) const noexcept = default;

	Position StartPosition() const noexcept { return start; }
	const std::vector<Field>& Moves() const noexcept { return moves; }
	std::vector<Position> Positions() const;

	void Play(Field move) { moves.push_back(move); }
};

std::string to_string(const Game&);
