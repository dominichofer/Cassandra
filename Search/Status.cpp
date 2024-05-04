#include "Status.h"

Status::Status(int fail_low_limit) noexcept
	: fail_low_limit(fail_low_limit)
	, best_score(-inf_score)
	, best_move(Field::PS)
	, lowest_intensity(64)
{}

void Status::Update(const Result& result, Field move)
{
	lowest_intensity = std::min(lowest_intensity, result.intensity);
	if (result.window.upper > best_score)
	{
		best_score = result.window.upper;
		best_move = move;
	}
}

Result Status::GetResult()
{
	if (best_score > fail_low_limit)
		return Result::Exact(best_score, lowest_intensity + 1, best_move);
	else
		return Result::FailLow(best_score, lowest_intensity + 1, best_move);
}
