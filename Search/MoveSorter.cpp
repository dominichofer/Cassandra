#include "MoveSorter.h"
#include "Algorithm.h"
#include <algorithm>

SortedMoves::SortedMoves(Moves possible_moves, std::function<uint32_t(Field)> metric)
{
	m_moves.reserve(possible_moves.size());
	for (Field move : possible_moves)
		m_moves.push_back((metric(move) << 8) + std::to_underlying(move));
	std::ranges::sort(m_moves, std::greater<uint32_t>());
}

int double_corner_popcount(uint64_t b)
{
	return std::popcount(b) + std::popcount(b & 0x8100000000000081ULL);
}

SortedMoves MoveSorter::Sorted(const Position& pos, Intensity intensity)
{
	Field tt_move = Field::PS;
	int8_t sort_depth = (intensity.depth - 16) / 2;
	if (auto look_up = tt.LookUp(pos); look_up.has_value())
		tt_move = look_up.value().best_move;

	auto metric = [&](Field move) -> uint32_t
		{
			if (move == tt_move)
				return 0x800000U;

			Position next = Play(pos, move);
			uint64_t O = next.Opponent();
			uint64_t E = next.Empties();

			uint32_t score = 0;
			score += (36 - double_corner_popcount(EightNeighboursAndSelf(O) & E)) << 4; // potential mobility, with corner bonus
			score += std::popcount(StableEdges(next) & O) << 10;
			score += (36 - double_corner_popcount(PossibleMoves(next))) << 15; // possible moves, with corner bonus
			if (sort_depth >= 0)
				score += (32 - alg.Eval(next, { sort_depth, intensity.level }).GetScore()) << 15;
			return score;
		};
	return SortedMoves(PossibleMoves(pos), metric);
}
