#pragma once
#include "AlphaBetaFailSoftSearch.h"
#include "Core/Moves.h"
#include "Core/Position.h"
#include "Core/Search.h"
#include "Engine/HashTablePVS.h"
#include <optional>

namespace Search
{
	Result StableStonesAnalysis(const Position&);

	class Limits
	{
	public: // TODO: Make it private!
		const Intensity& requested;
		ClosedInterval possible = ClosedInterval::Whole();
		unsigned int worst_depth = 64;
		Selectivity worst_selectivity = Selectivity::None;
		Field best_move = Field::invalid;
		std::size_t node_count = 1;

	public:
		Limits() = delete;
		Limits(const Intensity& requested) noexcept : requested(requested) {}

		void Improve(const Result&);
		void Improve(const std::optional<Result>&);

		bool HasResult() const;
		Result GetResult() const { return { possible, worst_depth, worst_selectivity, best_move, node_count }; }
	};

	class StatusQuo
	{
		const Intensity& requested;
		OpenInterval searching;
		unsigned int worst_depth = 64;
		Selectivity worst_selectivity = Selectivity::None;
		std::size_t node_count = 1;
		ClosedInterval best_interval = ClosedInterval::Whole();
		Field best_move = Field::invalid;
		ClosedInterval possible;

	public:
		StatusQuo() = delete;
		StatusQuo(const Limits&) noexcept;

		Intensity NextPvsIntensity() const noexcept;
		Intensity NextZwsIntensity() const noexcept;
		OpenInterval SearchWindow() const noexcept { return searching; }

		bool Improve(const Result& novum, Field move);

		bool IsUpperCut() const noexcept { return best_interval > searching; }
		Result GetResult() const noexcept {
			return { Overlap(best_interval, possible), worst_depth, worst_selectivity, best_move, node_count };
		}
	};

	// Transposition table updater
	template <typename TT>
	class TT_Updater
	{
		const Position& pos;
		TT& tt; // Transposition table
		const StatusQuo& status_quo;
	public:
		TT_Updater(const Position& pos, TT& tt, const StatusQuo& status_quo) : pos(pos), tt(tt), status_quo(status_quo) {}
		~TT_Updater() { tt.Update(pos, status_quo.GetResult()); }
	};

	class PVSearch : public Algorithm
	{
		HashTablePVS& tt;
	public:
		PVSearch(HashTablePVS& tt) : tt(tt) {}

		using Algorithm::Eval;
		Result Eval(Position, Intensity requested) override;
	private:
		Result PVS_N(const Position&, const Intensity&);
		Result ZWS_N(const Position&, const Intensity&);
		Result ZWS_A(const Position&, const Intensity&);
	};
}
