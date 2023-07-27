#pragma once
#include "Board/Board.h"
#include "Score.h"
#include <string>
#include <string_view>

struct PosScore
{
	Position pos;
	int score = undefined_score;

	bool operator==(const PosScore&) const noexcept = default;
	bool operator!=(const PosScore&) const noexcept = default;
};

bool IsPositionScore(std::string_view);
std::string to_string(const PosScore&);
PosScore PosScoreFromString(std::string_view);
