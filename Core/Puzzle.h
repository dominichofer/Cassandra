#pragma once
#include "Position.h"
#include "Search.h"

class Puzzle
{
	Position position;
	Search::Intensity intensity;
	Search::Result result;
public:
	Puzzle(Position, Search::Intensity, Search::Result = { 0, Field::invalid, 0 });

	auto Position() const { return position; }
	auto Intensity() const { return intensity; }
	auto Result() const { return result; }

	void Solve(Search::Algorithm&);
};
