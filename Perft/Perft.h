#pragma once
#include "Board/Board.h"
#include "Hashtable.h"
#include <cstdint>

int64_t Correct(int depth);

class Perft
{
protected:
	HashTable& tt;
	const bool cuda;
	const int initial_unroll = 6;
	const int cuda_depth = 8;

	int64_t calculate_1(const Position&) const;
	int64_t calculate_2(const Position&) const;
	int64_t calculate_n(Position, int depth);
public:
	Perft(HashTable& tt, bool cuda = false) : tt(tt), cuda(cuda) {}

	int64_t calculate(const Position&, int depth);
	int64_t calculate(int depth);
};
