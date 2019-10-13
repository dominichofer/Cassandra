#include "Search.h"
#include "Machine.h"
#include <limits>

using namespace Search;

Score EvalGameOver(Position pos)
{
	const auto Ps = static_cast<Score>(PopCount(pos.GetP()));
	const auto Os = static_cast<Score>(PopCount(pos.GetO()));
	if (Ps > Os)
		return 64 - 2 * Os;
	if (Ps < Os)
		return 2 * Ps - 64;
	return Ps - Os;
}

Search::Window::Window(Score lower, Score upper)
	: lower(lower), upper(upper)
{
	assert(-infinity <= lower);
	assert(lower <= +infinity);

	assert(-infinity <= upper);
	assert(upper <= +infinity);

	assert(lower <= upper);
}

Selectivity::Selectivity(float quantile) : quantile(quantile)
{
	assert((quantile >= 0) || quantile == None.quantile);
}

const Selectivity Selectivity::None = Selectivity(std::numeric_limits<decltype(Selectivity::quantile)>::infinity());

Intensity Intensity::Exact(Position pos)
{
	return { Window{}, static_cast<unsigned int>(pos.EmptyCount()), Selectivity::None };
}

Result Search::Result::ExactScore(Score score, unsigned int depth, Selectivity selectivity, Field best_move, std::size_t node_count)
{
	return { { score, score }, depth, selectivity, best_move, node_count };
}

Result Search::Result::MaxBound(Score score, unsigned int depth, Selectivity selectivity, Field best_move, std::size_t node_count)
{
	return { { -infinity, score }, depth, selectivity, best_move, node_count };
}

Result Search::Result::MinBound(Score score, unsigned int depth, Selectivity selectivity, Field best_move, std::size_t node_count)
{
	return { { score, +infinity }, depth, selectivity, best_move, node_count };
}
