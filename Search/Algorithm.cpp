#include "Algorithm.h"
#include <algorithm>

//OpenInterval NextZeroWindow(const OpenInterval& window, int best_score) noexcept
//{
//	int lower = std::max(window.Lower(), best_score);
//	return OpenInterval{-lower - 1, -lower};
//}
//
//OpenInterval NextFullWindow(const OpenInterval& window, int best_score) noexcept
//{
//	int lower = std::max(window.Lower(), best_score);
//	return OpenInterval{ -window.Upper(), -lower };
//}
//
//Result AllMovesSearched(const OpenInterval& window, const Findings& findings) noexcept
//{
//	int score = findings.best_score;
//	if (score < window) // Failed low
//		return Result::FailLow(findings.lowest_intensity, score);
//	return Result(findings.lowest_intensity, score);
//}
//
//void Findings::Add(const IntensityInterval& result, Field move) noexcept
//{
//	if (result.Upper() > best_score)
//	{
//		best_score = result.Upper();
//		best_move = move;
//	}
//	lowest_intensity = std::min(lowest_intensity, result.intensity);
//}
//
//uint64_t PotentialMoves(const Position& pos) noexcept
//{
//	return EightNeighboursAndSelf(pos.Opponent()) & pos.Empties();
//	//BitBoard O = pos.Opponent();
//	//O |= (O >> 8) | (O << 8);
//	//BitBoard tmp = O & 0x7E7E7E7E7E7E7E7EULL;
//	//O |= tmp << 1 | tmp >> 1;
//	//return O & pos.Empties();
//}
//
//uint64_t OpponentsExposed(const Position& pos) noexcept
//{
//	auto b = pos.Empties();
//	b |= ((b >> 1) & 0x7F7F7F7F7F7F7F7FULL) | ((b << 1) & 0xFEFEFEFEFEFEFEFEULL);
//	b |= (b >> 8) | (b << 8);
//	return b & pos.Opponent();
//}
//
////static const int8_t FieldValue[64] = {
////	9, 2, 8, 6, 6, 8, 2, 9,
////	2, 1, 3, 4, 4, 3, 1, 2,
////	8, 3, 7, 5, 5, 7, 3, 8,
////	6, 4, 5, 0, 0, 5, 4, 6,
////	6, 4, 5, 0, 0, 5, 4, 6,
////	8, 3, 7, 5, 5, 7, 3, 8,
////	2, 1, 3, 4, 4, 3, 1, 2,
////	9, 2, 8, 6, 6, 8, 2, 9,
////};
//
//int32_t MoveOrderingScorer(const Position& pos, Field move) noexcept
//{
//	Position next = Play(pos, move);
//	auto pm = PossibleMoves(next);
//
//	int score = 0; // FieldValue[static_cast<uint8_t>(move)];
//	score -= popcount(EightNeighboursAndSelf(next.Empties()) & next.Opponent()) * 0.36; // w_potential_mobility
//	score += popcount(EightNeighboursAndSelf(next.Player()) & next.Empties()) * 0.32;
//	score -= popcount(EightNeighboursAndSelf(next.Opponent()) & next.Empties()) * 0.29;
//	score += popcount(StableEdges(next) & next.Opponent()) * 0.55; // w_edge_stability
//	score -= pm.size() * 0.76; // w_mobility
//	score -= (pm & BitBoard::Corners()).size() * 0.84; // w_mobility
//	return score;
//
//	//auto next_pos = Play(pos, move);
//	//auto next_possible_moves = PossibleMoves(next_pos);
//	//auto mobility_score = next_possible_moves.size() << 17;
//	//next_possible_moves.Filter(BitBoard::Corners());
//	//auto corner_mobility_score = next_possible_moves.size() << 18;
//
//	//auto potential_moves = PotentialMoves(next_pos);
//	//auto potential_moves_score = (popcount(potential_moves) + popcount(potential_moves & BitBoard::Corners())) << 5;
//
//	//auto field_score = FieldValue[static_cast<uint8_t>(move)];
//	//return field_score - mobility_score - corner_mobility_score - potential_moves_score;
//}

int Algorithm::Eval(const Position& pos, Intensity depth)
{
	return Eval(pos, depth, { -inf_score, +inf_score });
}

int Algorithm::Eval(const Position& pos, OpenInterval window)
{
	return Eval(pos, pos.EmptyCount(), window);
}

int Algorithm::Eval(const Position& pos)
{
	return Eval(pos, pos.EmptyCount());
}

ScoreMove Algorithm::Eval_BestMove(const Position& pos, Intensity depth)
{
	return Eval_BestMove(pos, depth, { -inf_score, +inf_score });
}

ScoreMove Algorithm::Eval_BestMove(const Position& pos, OpenInterval window)
{
	return Eval_BestMove(pos, pos.EmptyCount(), window);
}

ScoreMove Algorithm::Eval_BestMove(const Position& pos)
{
	return Eval_BestMove(pos, pos.EmptyCount());
}

void ScoreMove::ImproveWith(int score, Field move)
{
	if (score > this->score)
	{
		this->score = score;
		this->move = move;
	}
}
