#pragma once
#include "Core/Position.h"

std::size_t Correct(int depth);

namespace Basic
{
	std::size_t perft(Position, int depth);
	std::size_t perft(int depth);
}

namespace Unrolled2
{
	std::size_t perft(Position, int depth);
	std::size_t perft(int depth);
}

namespace HashTableMap
{
	std::size_t perft(Position, int depth, std::size_t BytesRAM);
	std::size_t perft(int depth, std::size_t BytesRAM);
}