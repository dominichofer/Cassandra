#pragma once
#include "Core/Core.h"
#include "Algorithm.h"
#include "NegaMax.h"

class AlphaBetaFailSoft : public NegaMax
{
public:
	using NegaMax::Eval;
	ContextualResult Eval(const Position&, Intensity, OpenInterval) override;
	ContextualResult Eval(const Position&, OpenInterval);
private:
	int Eval_2(const Position&, OpenInterval, Field, Field);
	int Eval_3(const Position&, OpenInterval, Field, Field, Field);
protected:
	int Eval_N(const Position&, OpenInterval);
};
