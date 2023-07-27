#pragma once
#include "Board/Board.h"
#include <numeric>

// Estimates the score of a position and its accuracy
// Interface
class Estimator
{
public:
	virtual float Score(const Position&) const noexcept = 0;
	virtual float Accuracy(int empty_count, int small_depth, int big_depth) const noexcept = 0;
};

class EstimatorStub final : public Estimator
{
public:
	float Score(const Position&) const noexcept override { return 0; }
	float Accuracy(int empty_count, int small_depth, int big_depth) const noexcept override { return std::numeric_limits<float>::infinity(); }
};
