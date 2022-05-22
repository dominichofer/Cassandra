#include "pch.h"

bool AllTrue(const std::valarray<bool>& v)
{
	return std::ranges::all_of(v, std::identity());
}