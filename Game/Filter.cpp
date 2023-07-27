#include "Filter.h"

std::vector<Position> EmptyCountFiltered(std::span<const Position> pos, int min_empty_count, int max_empty_count)
{
	std::vector<Position> ret;
	for (const Position& p : pos)
		if (min_empty_count <= p.EmptyCount() and p.EmptyCount() <= max_empty_count)
			ret.push_back(p);
	return ret;
}

std::vector<Position> EmptyCountFiltered(std::span<const Position> pos, int empty_count)
{
	return EmptyCountFiltered(pos, empty_count, empty_count);
}

std::vector<PosScore> EmptyCountFiltered(std::span<const PosScore> pos_score, int min_empty_count, int max_empty_count)
{
	std::vector<PosScore> ret;
	for (const PosScore& ps : pos_score)
		if (min_empty_count <= ps.pos.EmptyCount() and ps.pos.EmptyCount() <= max_empty_count)
			ret.push_back(ps);
	return ret;
}

std::vector<PosScore> EmptyCountFiltered(std::span<const PosScore> pos_score, int empty_count)
{
	return EmptyCountFiltered(pos_score, empty_count, empty_count);
}