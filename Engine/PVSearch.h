#pragma once
#include "AlphaBetaFailSoftSearch.h"
#include "Core/Moves.h"
#include "Core/Position.h"
#include "Core/Search.h"
#include "Engine/HashTablePVS.h"
#include <optional>

namespace Search
{
	class PVSearch : public Algorithm
	{
		HashTablePVS& tt;

		class StatusQuo
		{
			Intensity intensity;
			Score best_score = -Score::Infinity;
			Field best_move = Field::invalid;
			unsigned int worst_depth;
			Selectivity worst_selectivity;
			std::size_t node_count = 1;
			std::optional<Result> result = std::nullopt;

		public:
			StatusQuo() = delete;
			StatusQuo(Intensity required) noexcept : intensity(required), worst_depth(required.depth), worst_selectivity(required.selectivity) {}

			operator Intensity() const noexcept;
			ExclusiveInterval Window() const noexcept { return intensity.window; }

			void ImproveWithMove(const Result&, Field move);
			void ImproveWithAny(const Result&);
			void ImproveWithAny(const std::optional<Result>&);

			bool HasResult() const { return result.has_value(); }
			Result GetResult() const { return result.value(); }

			void AllMovesTried(const Intensity& requested);
		};

		class TT_Updater
		{
			const Position& pos;
			HashTablePVS& tt;
			const PVSearch::StatusQuo& status_quo;
		public:
			TT_Updater(const Position& pos, HashTablePVS& tt, const PVSearch::StatusQuo& status_quo) : pos(pos), tt(tt), status_quo(status_quo) {}
			~TT_Updater() { tt.Update(pos, status_quo.GetResult()); }
		};

	public:
		PVSearch(HashTablePVS& tt) : tt(tt) {}

		Result Eval(Position, Intensity);
	private:
		Result PVS_N(const Position&, const Intensity&);
		Result ZWS_N(const Position&, const Intensity&);
	};
}