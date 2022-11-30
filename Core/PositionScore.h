#pragma once
#include "Position.h"
#include <string>

struct PosScore
{
	Position pos;
	int score = undefined_score;

	bool operator==(const PosScore&) const noexcept = default;
	bool operator!=(const PosScore&) const noexcept = default;
};

std::string to_string(const PosScore&);

inline int EmptyCount(const PosScore& ps) { return ps.pos.EmptyCount(); }