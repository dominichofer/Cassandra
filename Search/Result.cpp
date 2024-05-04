#include "Result.h"
#include "Game/Game.h"
#include <cassert>
#include <format>

Result::Result(ClosedInterval window, Intensity intensity, Field best_move) noexcept
	: window(window)
	, intensity(intensity)
	, best_move(best_move)
{
	assert(window.lower <= window.upper);
	assert(window.lower >= min_score);
	assert(window.upper <= max_score);
}

Result Result::FailLow(Score score, Intensity intensity, Field best_move) noexcept
{
	return Result({ min_score, score }, intensity, best_move);
}

Result Result::Exact(Score score, Intensity intensity, Field best_move) noexcept
{
	return Result({ score, score }, intensity, best_move);
}

Result Result::FailHigh(Score score, Intensity intensity, Field best_move) noexcept
{
	return Result({ score, max_score }, intensity, best_move);
}

Result Result::operator-() const noexcept
{
	return Result{ -window, intensity, best_move };
}

Result Result::operator+(int depth) const noexcept
{
	return Result{ window, intensity + depth, best_move };
}

bool Result::IsExact() const noexcept
{
	return window.lower == window.upper;
}

Score Result::GetScore() const noexcept
{
	if (window.lower == min_score)
		return window.upper;
	else
		return window.lower;
}

std::string to_string(const Result& result)
{
	using std::to_string;
	return std::format("{} d{} {}",
		to_string(result.window),
		to_string(result.intensity),
		to_string(result.best_move));
}