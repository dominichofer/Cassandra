#pragma once
#include "Position.h"
#include "Search.h"
#include <optional>

class Puzzle
{
	Position position;
	Search::Intensity intensity;
	std::optional<Search::Result> result;
public:
	Puzzle(Position, Search::Intensity, Search::Result);
	Puzzle(Position, Search::Intensity);

	auto Position() const { return position; }
	auto Intensity() const { return intensity; }
	auto Result() const { return result; }

	void Solve(Search::Algorithm&);
};
