#include "Filter.h"

std::vector<Position> EmptyCountFiltered(std::span<const Position> pos, int lower, int upper)
{
	std::vector<Position> ret;
	for (const Position& p : pos)
		if (lower <= p.EmptyCount() and p.EmptyCount() <= upper)
			ret.push_back(p);
	return ret;
}

std::vector<Position> EmptyCountFiltered(std::span<const Position> pos, int empty_count)
{
	return EmptyCountFiltered(pos, empty_count, empty_count);
}

std::vector<ScoredPosition> EmptyCountFiltered(std::span<const ScoredPosition> pos_score, int lower, int upper)
{
	std::vector<ScoredPosition> ret;
	for (const ScoredPosition& ps : pos_score)
		if (lower <= ps.pos.EmptyCount() and ps.pos.EmptyCount() <= upper)
			ret.push_back(ps);
	return ret;
}

std::vector<ScoredPosition> EmptyCountFiltered(std::span<const ScoredPosition> pos_score, int empty_count)
{
	return EmptyCountFiltered(pos_score, empty_count, empty_count);
}