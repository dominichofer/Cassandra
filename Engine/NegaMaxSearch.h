#pragma once
#include "Core/Moves.h"
#include "Core/Position.h"
#include "Core/Search.h"

namespace Search
{
	class NegaMax : public Algorithm
	{
	public:		
		Result Eval(Position, Intensity) override;
	protected:
		std::size_t node_counter = 0;

		Score Eval_0(const Position&);
		Score Eval_1(const Position&, Field);
	private:
		Score Eval_2(const Position&, Field, Field);
		Score Eval_3(const Position&, Field, Field, Field);
		Score Eval_4(const Position&, Field, Field, Field, Field);
		Score Eval_N(const Position&);

		Score Eval_triage(const Position&);
	};
}
