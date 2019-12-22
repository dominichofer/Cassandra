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
	return 0;
}

//ExclusiveInterval::ExclusiveInterval(Score lower, Score upper) noexcept
//	: lower(lower), upper(upper)
//{
//	assert(-infinity <= lower);
//	assert(lower <= +infinity);
//
//	assert(-infinity <= upper);
//	assert(upper <= +infinity);
//
//	assert(lower <= upper);
//}
//
//bool ExclusiveInterval::Contains(Score s) const noexcept
//{
//	return (lower < s) && (s < upper);
//}
//
//bool ExclusiveInterval::IsAbove(Score s) const noexcept
//{
//	return (lower >= s);
//}
//
//bool ExclusiveInterval::IsBelow(Score s) const noexcept
//{
//	return (upper <= s);
//}
//
//bool ExclusiveInterval::IsAbove(ExclusiveInterval w) const noexcept
//{
//	return IsAbove(w.upper);
//}
//
//bool ExclusiveInterval::IsBelow(ExclusiveInterval w) const noexcept
//{
//	return IsBelow(w.lower);
//}
//
//const ExclusiveInterval ExclusiveInterval::Full = ExclusiveInterval{ -infinity, infinity };

Selectivity::Selectivity(float quantile) : quantile(quantile)
{
	assert((quantile >= 0) || quantile == None.quantile);
}

const Selectivity Selectivity::None = Selectivity(std::numeric_limits<decltype(Selectivity::quantile)>::infinity());
const Selectivity Selectivity::Infinit = Selectivity(0);

Intensity Intensity::Exact(Position pos)
{
	return { { -infinity, infinity }, static_cast<unsigned int>(pos.EmptyCount()), Selectivity::None };
}

Intensity Intensity::operator-() const
{
	return { -window, depth, selectivity };
}

Intensity Intensity::operator-(int d) const
{
	return { window, depth - d, selectivity };
}

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
	return { { -infinity, score }, depth, selectivity, best_move, node_count };
}
Result Result::MaxBound(Score score, Intensity intensity, Field best_move, std::size_t node_count)
{
	return MaxBound(score, intensity.depth, intensity.selectivity, best_move, node_count);
}

Result Result::MinBound(Score score, unsigned int depth, Selectivity selectivity, Field best_move, std::size_t node_count)
{
	return { { score, +infinity }, depth, selectivity, best_move, node_count };
}
Result Result::MinBound(Score score, Intensity intensity, Field best_move, std::size_t node_count)
{
	return MinBound(score, intensity.depth, intensity.selectivity, best_move, node_count);
}

Result Result::FromScore(Score score, ExclusiveInterval window, unsigned int depth, Selectivity selectivity, Field best_move, std::size_t node_count)
{
	if (window.Contains(score))
		return ExactScore(score, depth, selectivity, best_move, node_count);
	if (window > score)
		return MaxBound(score, depth, selectivity, best_move, node_count);
	if (window < score)
		return MinBound(score, depth, selectivity, best_move, node_count);
}
Result Result::FromScore(Score score, Intensity intensity, Field best_move, std::size_t node_count)
{
	return FromScore(score, intensity.window, intensity.depth, intensity.selectivity, best_move, node_count);
}

Result Result::operator-() const
{
	return { -window, depth, selectivity, best_move, node_count };
}

bool Result::Exceeds(ExclusiveInterval w) const noexcept
{
	return window > w;
}
