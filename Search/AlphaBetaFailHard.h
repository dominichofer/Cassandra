#pragma once
#include "Core/Core.h"
#include "Algorithm.h"
#include "NegaMax.h"

class AlphaBetaFailHard : public NegaMax
{
public:
	using NegaMax::Eval;
	using NegaMax::Eval_BestMove;
	int Eval(const Position&, Intensity, OpenInterval) override;
	ScoreMove Eval_BestMove(const Position&, Intensity, OpenInterval) override;
private:
	ScoreMove Eval_BestMove_N(const Position&, OpenInterval);
	int Eval_0(const Position&, OpenInterval);
	int Eval_1(const Position&, OpenInterval, Field);
	int Eval_2(const Position&, OpenInterval, Field, Field);
	int Eval_3(const Position&, OpenInterval, Field, Field, Field);
protected:
	int Eval_N(const Position&, OpenInterval);
};
