#pragma once
#include "Position.h"
#include "Score.h"
#include <string>

struct PosScore
{
	Position pos;
	int score = undefined_score;

	bool operator==(const PosScore&) const noexcept = default;
	bool operator!=(const PosScore&) const noexcept = default;
};
