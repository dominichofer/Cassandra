#pragma once
#include "Core/Core.h"
#include "Algorithm.h"

class NegaMax : public Algorithm
{
protected:
	static inline thread_local uint64 nodes;
public:
	uint64 Nodes() const override { return nodes; }

	using Algorithm::Eval;
	using Algorithm::Eval_BestMove;
	int Eval(const Position&, Intensity, OpenInterval) override;
	ScoreMove Eval_BestMove(const Position&, Intensity, OpenInterval) override;
private:
	ScoreMove Eval_BestMove_N(const Position&);
	int Eval_N(const Position&);
	int Eval_3(const Position&, Field, Field, Field);
	int Eval_2(const Position&, Field, Field);
protected:
	int Eval_1(const Position&, Field);
	int Eval_0(const Position&);
};