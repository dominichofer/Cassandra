#pragma once
#include "Game/Game.h"
#include <string>

struct Result
{
	ClosedInterval window{ min_score, max_score };
	Intensity intensity{ -1, 0.0f };
	Field best_move = Field::PS;

	Result() noexcept = default;
	Result(ClosedInterval window, Intensity intensity, Field best_move) noexcept;
	static Result FailLow(Score, Intensity, Field best_move) noexcept;
	static Result Exact(Score, Intensity, Field best_move) noexcept;
	static Result FailHigh(Score, Intensity, Field best_move) noexcept;

	bool operator==(const Result&) const noexcept = default;
	bool operator!=(const Result&) const noexcept = default;
	Result operator-() const noexcept;
	Result operator+(int depth) const noexcept;

	bool IsExact() const noexcept;

	Score GetScore() const noexcept;
};

std::string to_string(const Result&);
