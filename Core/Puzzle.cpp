#include "Puzzle.h"
#include <cassert>

Puzzle::Puzzle(::Position position, Search::Intensity intensity, Search::Result result)
	: position(position)
	, intensity(intensity)
	, result(result)
{
	assert(intensity.depth <= position.EmptyCount());
}

Puzzle::Puzzle(::Position position, Search::Intensity intensity)
	: position(position)
	, intensity(intensity)
	, result({})
{
	assert(intensity.depth <= position.EmptyCount());
}

Puzzle Puzzle::Exact(::Position pos)
{
	return Puzzle(pos, Search::Intensity::Exact(pos));
}

bool Puzzle::IsSolved() const
{
	if (result)
		return (result.value().depth >= intensity.depth)
			&& (result.value().selectivity <= intensity.selectivity);
	return false;
}

void Puzzle::Reset()
{
	result = std::nullopt;
}

void Puzzle::Solve(Search::Algorithm& algorithm, bool force)
{
	if (force || !IsSolved())
		result = algorithm.Eval(position, intensity);
}
