#pragma once
#include "Core/Moves.h"
#include "Core/Position.h"
#include "Core/Search.h"
#include "NegaMaxSearch.h"

namespace Search
{
	class AlphaBetaFailSoft : public NegaMax
	{
	public:
		Result Eval(Position, Intensity) override;
	protected:
		Score Eval_2(const Position&, ExclusiveInterval, Field, Field);
		Score Eval_3(const Position&, ExclusiveInterval, Field, Field, Field);
		Score Eval_4(const Position&, ExclusiveInterval, Field, Field, Field, Field);
	private:
		Score Eval_triage(const Position&, ExclusiveInterval);
		Score Eval_N(const Position&, ExclusiveInterval);
	};
}
