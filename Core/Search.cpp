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

Selectivity::Selectivity(float quantile) : quantile(quantile)
{
	assert((quantile >= 0) || quantile == None.quantile);
}

const Selectivity Selectivity::None = Selectivity(std::numeric_limits<decltype(Selectivity::quantile)>::infinity());

Intensity Intensity::Exact(Position pos)
{
	return { static_cast<unsigned int>(pos.EmptyCount()), Selectivity::None, Window{} };
}
