#include "Algorithm.h"
#include <limits>

using namespace Search;

const Selectivity Selectivity::None = Selectivity(std::numeric_limits<decltype(Selectivity::quantile)>::infinity());
const Selectivity Selectivity::Infinit = Selectivity(0);

Intensity Intensity::Exact(Position pos)
{
	return { OpenInterval::Whole(), static_cast<unsigned int>(pos.EmptyCount()), Selectivity::None };
}

Intensity Intensity::operator-() const
{
	return { -window, depth, selectivity };
}

Intensity Intensity::operator-(int d) const
{
	return { window, depth - d, selectivity };
}

Intensity Intensity::next() const
{
	return { -window, depth - 1, selectivity };
}

Result::Result(ClosedInterval window, unsigned int depth, Selectivity selectivity, Field best_move, std::size_t node_count)
	: window(window), depth(depth), selectivity(selectivity), best_move(best_move), node_count(node_count)
{}

Result Result::ExactScore(Score score, unsigned int depth, Selectivity selectivity, Field best_move, std::size_t node_count)
{
	return { { score, score }, depth, selectivity, best_move, node_count };
}
Result Result::ExactScore(Score score, Intensity intensity, Field best_move, std::size_t node_count)
{
	return ExactScore(score, intensity.depth, intensity.selectivity, best_move, node_count);
}

Result Result::MaxBound(Score score, unsigned int depth, Selectivity selectivity, Field best_move, std::size_t node_count)
{
	return { { min_score, score }, depth, selectivity, best_move, node_count };
}
Result Result::MaxBound(Score score, Intensity intensity, Field best_move, std::size_t node_count)
{
	return MaxBound(score, intensity.depth, intensity.selectivity, best_move, node_count);
}

Result Result::MinBound(Score score, unsigned int depth, Selectivity selectivity, Field best_move, std::size_t node_count)
{
	return { { score, max_score }, depth, selectivity, best_move, node_count };
}
Result Result::MinBound(Score score, Intensity intensity, Field best_move, std::size_t node_count)
{
	return MinBound(score, intensity.depth, intensity.selectivity, best_move, node_count);
}

Result Result::operator-() const
{
	return { -window, depth, selectivity, best_move, node_count };
}
