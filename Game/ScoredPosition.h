#pragma once
#include "Board/Board.h"
#include "Score.h"
#include <string>
#include <string_view>

struct ScoredPosition
{
	Position pos;
	Score score = undefined_score;

	static ScoredPosition FromString(std::string_view);

	bool operator==(const ScoredPosition&) const noexcept = default;
	bool operator!=(const ScoredPosition&) const noexcept = default;

	int EmptyCount() const noexcept;
	bool HasScore() const noexcept;
};

std::string to_string(const ScoredPosition&);
