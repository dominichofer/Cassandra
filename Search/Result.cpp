#include "Result.h"
#include <limits>

ResultType operator-(ResultType type)
{
	return static_cast<ResultType>(-static_cast<int8_t>(type));
}

Result::Result(ResultType score_type, int8_t score, int8_t depth, float confidence_level, Field best_move) noexcept
	: score_type(score_type)
	, score(score)
	, depth(depth)
	, confidence_level(confidence_level)
	, best_move(best_move)
{}

Result Result::FailLow(int8_t score, int8_t depth, float confidence_level, Field best_move) noexcept
{
	return Result(ResultType::fail_low, score, depth, confidence_level, best_move);
}

Result Result::Exact(int8_t score, int8_t depth, float confidence_level, Field best_move) noexcept
{
	return Result(ResultType::exact, score, depth, confidence_level, best_move);
}

Result Result::FailHigh(int8_t score, int8_t depth, float confidence_level, Field best_move) noexcept
{
	return Result(ResultType::fail_high, score, depth, confidence_level, best_move);
}

Result Result::operator-() const noexcept
{
	return Result(-score_type, -score, depth, confidence_level, best_move);
}

bool Result::IsFailLow() const noexcept
{
	return score_type == ResultType::fail_low;
}

bool Result::IsExact() const noexcept
{
	return score_type == ResultType::exact;
}

bool Result::IsFailHigh() const noexcept
{
	return score_type == ResultType::fail_high;
}

ClosedInterval Result::Window() const noexcept
{
	if (IsFailLow())
		return ClosedInterval(min_score, score);
	if (IsExact())
		return ClosedInterval(score, score);
	if (IsFailHigh())
		return ClosedInterval(score, max_score);
}

Result Result::BetaCut(Field move) const noexcept
{
	return Result::FailHigh(-score, depth + 1, confidence_level, move);
}

std::string to_string(const Result& r)
{
	using std::to_string;
	return to_string(r.Window()) + " d" + to_string(r.depth) + "@" + to_string(r.confidence_level); // TODO: Add best_move!
}