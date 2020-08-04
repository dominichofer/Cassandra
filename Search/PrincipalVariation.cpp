#include "PrincipalVariation.h"
#include "SortedMoves.h"
#include "Core/Core.h"
#include <algorithm>

using namespace Search;

Result Search::StableStonesAnalysis(const Position& pos)
{
	auto opponent_stables = popcount(StableStones(pos));
	auto window = ClosedInterval(min_score, max_score - 2*opponent_stables);
	return Result(window, pos.EmptyCount(), Selectivity::None, Field::invalid, 0 /*nodes*/);
}

void Search::Limits::Improve(const Result& novum)
{
	if ((novum.depth < requested.depth) || (novum.selectivity > requested.selectivity))
		return; // novum uninteresting

	assert(possible.Overlaps(novum.window));

	if (novum.window.Contains(possible))
		return; // the possible window does not benefit from novum.

	worst_depth = std::min(worst_depth, novum.depth);
	worst_selectivity = std::max(worst_selectivity, novum.selectivity);
	//node_count += novum.node_count; // TODO: Uncomment!

	if (possible.Contains(novum.window))
		best_move = novum.best_move;
	else
		best_move = Field::invalid;

	possible = Overlap(possible, novum.window);
}

void Search::Limits::Improve(const std::optional<Result>& novum)
{
	if (novum.has_value())
		Improve(novum.value());
}

bool Search::Limits::HasResult() const
{
	return (possible.IsSingleton()) // exact score found
		|| (possible > requested.window) // above window of interest
		|| (possible < requested.window); // below window of interest
}



Search::StatusQuo::StatusQuo(const Limits& limits) noexcept
	: requested(limits.requested)
	, searching(limits.requested.window)
	, possible(limits.possible)
{
	if (limits.possible.Contains(limits.requested.window))
		return; // The requested window does not benefit from the limits.

	searching = Overlap(limits.requested.window, OpenInterval(limits.possible));
	worst_depth = limits.worst_depth;
	worst_selectivity = limits.worst_selectivity;
	node_count = limits.node_count;
}

Intensity Search::StatusQuo::NextPvsIntensity() const noexcept
{
	return Intensity(-searching, requested.depth - 1, requested.selectivity);
}

Intensity Search::StatusQuo::NextZwsIntensity() const noexcept
{
	return Intensity(-OpenInterval(searching.lower(), searching.lower()+1), requested.depth - 1, requested.selectivity);
}

bool StatusQuo::Improve(const Result& novum, Field move)
{
	worst_depth = std::min(worst_depth, novum.depth + 1);
	worst_selectivity = std::max(worst_selectivity, novum.selectivity);
	node_count += novum.node_count;

	if (novum.window > searching) // upper cut
	{
		best_interval = ClosedInterval(novum.window.lower(), max_score);
		best_move = move;
		return true;
	}

	if (best_interval == ClosedInterval::Whole())
	{
		best_interval = novum.window;
		best_move = move;
	}
	else
	{
		if (novum.window.upper() <= best_interval.lower())
			;
		else if (novum.window.lower() >= best_interval.upper())
			best_move = move;
		else
			best_move = Field::invalid;

		best_interval.try_increase_upper(novum.window.upper());
		best_interval.try_increase_lower(novum.window.lower());
	}

	return searching.try_increase_lower(novum.window.lower());
}

inline uint64_t OpponentsExposed(const Position& pos) noexcept
{
	auto b = pos.Empties();
	b |= ((b >> 1) & 0x7F7F7F7F7F7F7F7Fui64) | ((b << 1) & 0xFEFEFEFEFEFEFEFEui64);
	b |= (b >> 8) | (b << 8);
	return b & pos.O;
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

	const auto next_pos = Play(pos, move);
	const auto next_possible_moves = PossibleMoves(next_pos);
	const auto mobility_score = next_possible_moves.size() << 17;
	const auto corner_mobility_score = ((next_possible_moves.contains(Field::A1) ? 1 : 0)
									  + (next_possible_moves.contains(Field::A8) ? 1 : 0) 
									  + (next_possible_moves.contains(Field::H1) ? 1 : 0) 
									  + (next_possible_moves.contains(Field::H8) ? 1 : 0)) << 18;
	const auto opponents_exposed_score = popcount(OpponentsExposed(next_pos)) << 6;
	const auto field_score = FieldValue[static_cast<uint8_t>(move)];
	return field_score - mobility_score - corner_mobility_score - opponents_exposed_score;
}


Result PVSearch::Eval(Position pos, Intensity requested)
{
	return PVS_N(pos, requested);
}

Result PVSearch::PVS_N(const Position& pos, const Intensity& requested)
{
	if (pos.EmptyCount() <= 4)
		return AlphaBetaFailSoft{}.Eval(pos, requested);
	
	Moves moves = PossibleMoves(pos);
	if (moves.empty())
	{
		const auto passed = PlayPass(pos);
		if (PossibleMoves(passed).empty())
			return Result::ExactScore(EvalGameOver(pos), requested, Field::invalid, 1 /*node_count*/);
		return -PVS_N(passed, -requested);
	}

	Limits limits(requested);

	limits.Improve(StableStonesAnalysis(pos));
	limits.Improve(tt.LookUp(pos));
	if (limits.HasResult())
		return limits.GetResult();

	StatusQuo status_quo(limits);
	TT_Updater tt_updater(pos, tt, status_quo);

	bool first = true;
	SortedMoves sorted_moves(pos, [&](Field move) { return MoveOrderingScorer(pos, move); });
	for (auto move : sorted_moves)
	{
		if (!first)
		{
			const auto result = -ZWS_N(Play(pos, move.second), status_quo.NextZwsIntensity());
			if (result.window < status_quo.SearchWindow())
			{
				status_quo.Improve(result, move.second);
				continue;
			}
			if (result.window > status_quo.SearchWindow())
			{
				status_quo.Improve(result, move.second);
				if (status_quo.IsUpperCut())
					return status_quo.GetResult();
			}
		}
		first = false;

		const auto result = -PVS_N(Play(pos, move.second), status_quo.NextPvsIntensity());
		status_quo.Improve(result, move.second);
		if (status_quo.IsUpperCut())
			return status_quo.GetResult();
	}

	return status_quo.GetResult();
}

Result PVSearch::ZWS_N(const Position& pos, const Intensity& requested)
{
	if (pos.EmptyCount() <= 4)
		return AlphaBetaFailSoft{}.Eval(pos, requested);
	if (pos.EmptyCount() <= 7)
		return ZWS_A(pos, requested);

	Moves moves = PossibleMoves(pos);
	if (moves.empty())
	{ 
		const auto passed = PlayPass(pos);
		if (PossibleMoves(passed).empty())
			return Result::ExactScore(EvalGameOver(pos), requested, Field::invalid, 1 /*node_count*/);
		return -ZWS_N(passed, -requested);
	}

	Limits limits(requested);

	limits.Improve(StableStonesAnalysis(pos));
	limits.Improve(tt.LookUp(pos));
	if (limits.HasResult())
		return limits.GetResult();

	StatusQuo status_quo(limits);
	TT_Updater tt_updater(pos, tt, status_quo);

	SortedMoves sorted_moves(pos, [&](Field move) { return MoveOrderingScorer(pos, move); });
	for (auto move : sorted_moves)
	{
		const auto result = -ZWS_N(Play(pos, move.second), status_quo.NextPvsIntensity());
		status_quo.Improve(result, move.second);
		if (status_quo.IsUpperCut())
			return status_quo.GetResult();
	}

	return status_quo.GetResult();
}

Result PVSearch::ZWS_A(const Position& pos, const Intensity& requested)
{
	if (pos.EmptyCount() <= 4)
		return AlphaBetaFailSoft{}.Eval(pos, requested);

	Moves moves = PossibleMoves(pos);
	if (moves.empty())
	{ 
		const auto passed = PlayPass(pos);
		if (PossibleMoves(passed).empty())
			return Result::ExactScore(EvalGameOver(pos), requested, Field::invalid, 1 /*node_count*/);
		return -ZWS_N(passed, -requested);
	}

	StatusQuo status_quo(requested);
	Moves parity_moves = moves;
	parity_moves.Filter(pos.ParityQuadrants());
	for (auto move : parity_moves)
	{
		const auto result = -ZWS_N(Play(pos, move), status_quo.NextPvsIntensity());
		status_quo.Improve(result, move);
		if (status_quo.IsUpperCut())
			return status_quo.GetResult();
	}

	Moves non_parity_moves = moves;
	non_parity_moves.Filter(~pos.ParityQuadrants());
	for (auto move : non_parity_moves)
	{
		const auto result = -ZWS_N(Play(pos, move), status_quo.NextPvsIntensity());
		status_quo.Improve(result, move);
		if (status_quo.IsUpperCut())
			return status_quo.GetResult();
	}

	return status_quo.GetResult();
}
