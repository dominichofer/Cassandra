#pragma once
#include "Position.h"

// Estimates the score of a position and its accuracy
class Estimator
{
	// Interface
public:
	virtual int Score(const Position&) const noexcept = 0;
	virtual float Accuracy(const Position&, int small_depth, int big_depth) const noexcept = 0;
};
