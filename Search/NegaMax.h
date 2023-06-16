#pragma once
#include "Core/Core.h"
#include <chrono>
#include <cstdint>

struct ScoreTimeNodes
{
	int score;
	std::chrono::duration<double> time;
	uint64_t nodes;
};

class NegaMax
{
protected:
	static inline thread_local uint64_t nodes;
public:
	ScoreTimeNodes Eval(const Position&);
protected:
	int Eval_N(const Position&);
	int Eval_3(const Position&, Field, Field, Field);
	int Eval_2(const Position&, Field, Field);
	int Eval_1(const Position&, Field);
	int Eval_0(const Position&);
};
