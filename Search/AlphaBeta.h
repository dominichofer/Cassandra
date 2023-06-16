#pragma once
#include "Core/Core.h"
#include "NegaMax.h"

// Alaph Beta fail soft search
class AlphaBeta : public NegaMax
{
public:
	ScoreTimeNodes Eval(const Position&, OpenInterval window = { -inf_score, +inf_score });
protected:
	int Eval_N(const Position&, OpenInterval window);
	int Eval_P(const Position&, OpenInterval window);
	int Eval_4(const Position&, OpenInterval window, Field, Field, Field, Field);
	int Eval_3(const Position&, OpenInterval window, Field, Field, Field);
	int Eval_2(const Position&, OpenInterval window, Field, Field);
};
