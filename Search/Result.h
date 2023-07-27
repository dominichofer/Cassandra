#pragma once
#include "Board/Board.h"
#include <cstdint>
#include <string>

enum class ResultType : int8_t
{
	fail_low = -1,
	exact = 0,
	fail_high = +1
};

ResultType operator-(ResultType);


class Result
{
public:
	ResultType score_type;
	int8_t score;
	int8_t depth;
	float confidence_level;
	Field best_move;

	Result(ResultType, int8_t score, int8_t depth, float confidence_level, Field best_move) noexcept;

	static Result FailLow(int8_t score, int8_t depth, float confidence_level, Field best_move) noexcept;
	static Result Exact(int8_t score, int8_t depth, float confidence_level, Field best_move) noexcept;
	static Result FailHigh(int8_t score, int8_t depth, float confidence_level, Field best_move) noexcept;

	Result operator-() const noexcept;

	bool IsFailLow() const noexcept;
	bool IsExact() const noexcept;
	bool IsFailHigh() const noexcept;

	ClosedInterval<> Window() const noexcept;
	Result BetaCut(Field move) const noexcept;
};

std::string to_string(const Result&);