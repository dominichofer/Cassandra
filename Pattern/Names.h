#pragma once
#include "Base/Base.h"
#include <cstdint>
#include <vector>

constexpr uint64_t operator""_pattern(const char* c, std::size_t size)
{
	assert(size == 120);

	uint64_t b = 0;
	for (int y = 0; y < 8; y++)
		for (int x = 0; x < 8; x++)
		{
			char symbol = c[119 - 15 * y - 2 * x];
			if (symbol != '-')
				b |= 1ULL << (x + 8 * y);
		}
	return b;
}

namespace pattern
{
	inline const uint64_t L0 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"# # # # # # # #"_pattern;
	inline const uint64_t L1 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"# # # # # # # #"
		"- - - - - - - -"_pattern;
	inline const uint64_t L2 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"# # # # # # # #"
		"- - - - - - - -"
		"- - - - - - - -"_pattern;
	inline const uint64_t L3 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"# # # # # # # #"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"_pattern;
	inline const uint64_t D4 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - #"
		"- - - - - - # -"
		"- - - - - # - -"
		"- - - - # - - -"_pattern;
	inline const uint64_t D5 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - #"
		"- - - - - - # -"
		"- - - - - # - -"
		"- - - - # - - -"
		"- - - # - - - -"_pattern;
	inline const uint64_t D6 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - #"
		"- - - - - - # -"
		"- - - - - # - -"
		"- - - - # - - -"
		"- - - # - - - -"
		"- - # - - - - -"_pattern;
	inline const uint64_t D7 =
		"- - - - - - - -"
		"- - - - - - - #"
		"- - - - - - # -"
		"- - - - - # - -"
		"- - - - # - - -"
		"- - - # - - - -"
		"- - # - - - - -"
		"- # - - - - - -"_pattern;
	inline const uint64_t D8 =
		"- - - - - - - #"
		"- - - - - - # -"
		"- - - - - # - -"
		"- - - - # - - -"
		"- - - # - - - -"
		"- - # - - - - -"
		"- # - - - - - -"
		"# - - - - - - -"_pattern;
	inline const uint64_t L02X =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- # - - - - # -"
		"# # # # # # # #"_pattern;
	inline const uint64_t B4 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - # # # #"
		"- - - - # # # #"_pattern;
	inline const uint64_t B5 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - # # # # #"
		"- - - # # # # #"_pattern;
	inline const uint64_t B6 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - # # # # # #"
		"- - # # # # # #"_pattern;
	inline const uint64_t Q3 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - # # #"
		"- - - - - # # #"
		"- - - - - # # #"_pattern;
	inline const uint64_t C3 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - #"
		"- - - - - - # #"
		"- - - - - # # #"_pattern;
	inline const uint64_t C3p1 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - #"
		"- - - - - - - #"
		"- - - - - - # #"
		"- - - - # # # #"_pattern;
	inline const uint64_t C3p2 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - #"
		"- - - - - - - #"
		"- - - - - - - #"
		"- - - - - - # #"
		"- - - # # # # #"_pattern;
	inline const uint64_t Ep =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - # # # # - -"
		"# - # # # # - #"_pattern;
	inline const uint64_t Comet =
		"# - - - - - - -"
		"- # - - - - - -"
		"- - # - - - - -"
		"- - - # - - - -"
		"- - - - # - - -"
		"- - - - - # - -"
		"- - - - - - # #"
		"- - - - - - # #"_pattern;

	inline const std::vector<uint64_t> logistello{ // From "M. Buro, Logistello: A Strong Learning Othello Program , 19th Annual Conference Gesellschaft für Klassifikation e.V. (1995), Basel"
		D4, D5, D6, D7, D8, 
		L02X, L1, L2, L3,
		Q3, B5
	};

	inline const std::vector<uint64_t> edax{ // From https://github.com/abulmo/edax-reversi/blob/v4.4/src/eval.c#L38-L98
		D4, D5, D6, D7, D8,
		L02X, L1, L2, L3,
		Q3, C3p2, Ep
	};

	inline const std::vector<uint64_t> cassandra{
		D4, D5, D6, D7, Comet,
		L02X, L1, L2, L3,
		Q3, C3p2, B5
	};
}