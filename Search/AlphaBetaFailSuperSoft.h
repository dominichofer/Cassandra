#pragma once
#include "Core/Core.h"
#include "Algorithm.h"
#include "NegaMax.h"

class AlphaBetaFailSuperSoft : public NegaMax
{
	static constexpr int Eval_to_ParitySort = 7;
public:
	using NegaMax::Eval;
	using NegaMax::Eval_BestMove;
	int Eval(const Position&, Intensity, OpenInterval) override;
	ScoreMove Eval_BestMove(const Position&, Intensity, OpenInterval) override;
private:
	ScoreMove Eval_BestMove_N(const Position&, OpenInterval);
	int Eval_2(const Position&, OpenInterval, Field, Field);
	int Eval_3(const Position&, OpenInterval, Field, Field, Field);
	int Eval_P(const Position&, OpenInterval); // Parity based move sorting.
protected:
	int Eval_N(const Position&, OpenInterval);
};