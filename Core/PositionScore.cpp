#include "PositionScore.h"
#include "Format.h"

std::string to_string(const PosScore& ps)
{
	return fmt::format("{} % {:+03}", ps.pos, ps.score * 2);
}