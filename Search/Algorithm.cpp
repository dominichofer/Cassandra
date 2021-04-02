#include "Algorithm.h"
#include <algorithm>

using namespace Search;

Request NextZWS(const Request& request, const Findings& findings) noexcept
{
	auto lower = std::max(request.window.lower(), findings.best_score);
	return { {request.depth() - 1, request.certainty()}, -OpenInterval{lower, lower + 1} };
}

Request NextFWS(const Request& request, const Findings& findings) noexcept
{
	auto lower = std::max(request.window.lower(), findings.best_score);
	return { {request.depth() - 1, request.certainty()}, -OpenInterval{lower, request.window.upper()} };
}

Result AllMovesSearched(const Request& request, const Findings& findings) noexcept
{
	assert(findings.lowest_intensity >= request.intensity);
	int score = findings.best_score;
	if (score < request.window) // Failed low
		return Result::FailLow(findings.lowest_intensity, score);
	return Result::FoundScore(findings.lowest_intensity, score);
}

void Findings::Add(const Result& result, Field move) noexcept
{
	if (result.window.upper() > best_score)
	{
		best_score = result.window.upper();
		best_move = move;
	}
	lowest_intensity = std::min(lowest_intensity, result.intensity);
}

uint64_t OpponentsExposed(const Position& pos) noexcept
{
	auto b = pos.Empties();
	b |= ((b >> 1) & 0x7F7F7F7F7F7F7F7Fui64) | ((b << 1) & 0xFEFEFEFEFEFEFEFEui64);
	b |= (b >> 8) | (b << 8);
	return b & pos.Opponent();
}

int32_t MoveOrderingScorer(const Position& pos, Field move) noexcept
{
	static const int8_t FieldValue[64] = {
		9, 2, 8, 6, 6, 8, 2, 9,
		2, 1, 3, 4, 4, 3, 1, 2,
		8, 3, 7, 5, 5, 7, 3, 8,
		6, 4, 5, 0, 0, 5, 4, 6,
		6, 4, 5, 0, 0, 5, 4, 6,
		8, 3, 7, 5, 5, 7, 3, 8,
		2, 1, 3, 4, 4, 3, 1, 2,
		9, 2, 8, 6, 6, 8, 2, 9,
	};

	auto next_pos = Play(pos, move);
	auto next_possible_moves = PossibleMoves(next_pos);
	auto mobility_score = next_possible_moves.size() << 17;
	next_possible_moves.Filter(BitBoard::Corners());
	auto corner_mobility_score = next_possible_moves.size() << 18;
	auto opponents_exposed_score = popcount(OpponentsExposed(next_pos)) << 6;
	auto field_score = FieldValue[static_cast<uint8_t>(move)];
	return field_score - mobility_score - corner_mobility_score - opponents_exposed_score;
}

int32_t MoveOrderingScorer(const Position& pos, Field move, Field best_move) noexcept
{
	if (move == best_move)
		return 1 << 30;
	return MoveOrderingScorer(pos, move);
}