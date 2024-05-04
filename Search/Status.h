#pragma once
#include "Board/Board.h"
#include "Game/Game.h"
#include "Result.h"
#include <cmath>

class Status
{
	int fail_low_limit;
	int best_score;
	Field best_move;
	Intensity lowest_intensity;
public:
	Status(int fail_low_limit) noexcept;

	void Update(const Result&, Field move);
	Result GetResult();
};
