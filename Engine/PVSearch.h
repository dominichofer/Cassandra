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
			Score best_score = -infinity;
			Field best_move = Field::invalid;
			unsigned int worst_depth = 0;
			Selectivity worst_selectivity = Selectivity::Infinit;
			std::size_t node_count = 1;

		public:
			StatusQuo() = delete;
			StatusQuo(Intensity intensity) noexcept : intensity(intensity) {}

			operator Intensity() const noexcept;
			ExclusiveInterval Window() const noexcept { return intensity.window; }

			void ImproveWith(Result);
			void ImproveWith(const std::optional<PVS_Info>&);

			Result UpperCut(Result);
			Result AllMovesTried(Intensity);
		};

	public:
		PVSearch(HashTablePVS& tt) : tt(tt) {}

		Result Eval(Position, Intensity);
	private:
		Result PVS_N(const Position&, Intensity);
		Result ZWS_N(const Position&, Intensity);
	};
}