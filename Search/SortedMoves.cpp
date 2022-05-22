#include "SortedMoves.h"
#include "Core/Core.h"
#include "Stability.h"
#include <array>

SortedMoves MoveSorter::Sort(const Moves& moves, const Position& pos) const
{
	return SortedMoves(moves, [&](Field move)
		{
			Position next = Play(pos, move);
			auto P = next.Player();
			auto O = next.Opponent();
			auto E = next.Empties();
			auto pm = PossibleMoves(next);
			auto P9 = EightNeighboursAndSelf(P);
			auto O9 = EightNeighboursAndSelf(O);
			auto E9 = EightNeighboursAndSelf(E);

			std::array<std::function<BitBoard(BitBoard, BitBoard)>, 3> ops
			{
				[](BitBoard l, BitBoard r) { return popcount(l & r); },
				[](BitBoard l, BitBoard r) { return popcount(l | r); },
				[](BitBoard l, BitBoard r) { return popcount(l ^ r); }
			};
			int i = 0;
			float sum = 0;
			for (auto op : ops)
			{
				sum += (weights[i++] * op(P, O9));
				sum += (weights[i++] * op(P, E9));
				sum += (weights[i++] * op(O, P9));
				sum += (weights[i++] * op(O, E9));
				sum += (weights[i++] * op(E, P9));
				sum += (weights[i++] * op(E, O9));
				sum += (weights[i++] * op(P9, O9));
				sum += (weights[i++] * op(P9, E9));
				sum += (weights[i++] * op(O9, E9));
			}
			return static_cast<int>(std::round(sum));

			//int score = 0;
			//score -= DoubleCornerPopcount(PotentialMoves(next)) << a;
			//score -= popcount(EightNeighboursAndSelf(next.Empties()) & next.Opponent()) << b;
			//score -= DoubleCornerPopcount(PossibleMoves(next)) << c;
			//score += DoubleCornerPopcount(StableEdges(next) & next.Opponent()) << d;
			//return score;
		});
}