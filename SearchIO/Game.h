#pragma once
#include "Core/Core.h"
#include "IO/IO.h"

inline std::string to_string(const Game& game)
{
	return fmt::format("{}", fmt::join(game.moves, " "));
}
