#include "Objects.h"

using namespace Search;

Result Result::FoundScore(const Intensity& intensity, int score) noexcept
{
	return { intensity, {score, score} };
}

Result Result::FailHigh(const Intensity& intensity, int score) noexcept
{
	return { intensity, {score, max_score} };
}

Result Result::FailLow(const Intensity& intensity, int score) noexcept
{
	return { intensity, {min_score, score} };
}

Result Result::FailHigh(const Result& result) noexcept
{
	return FailHigh(result.intensity, result.window.lower());
}

Result Result::FailLow(const Result& result) noexcept
{
	return FailLow(result.intensity, result.window.upper());
}

Result Result::Certain(int depth, int score) noexcept
{
	return { Intensity::Certain(depth), {score, score} }; // TODO: Make use of FoundScore etc.
}

Result Result::CertainFailHigh(int depth, int score) noexcept
{
	return { Intensity::Certain(depth), {score, max_score} };
}

Result Result::CertainFailLow(int depth, int score) noexcept
{
	return { Intensity::Certain(depth), {min_score, score} };
}

Result Result::CertainFailSoft(const Request& request, int depth, int score) noexcept
{
	if (score > request)
		return Result::CertainFailHigh(depth, score);
	if (score < request)
		return Result::CertainFailLow(depth, score);
	return Result::Certain(depth, score);
}

Result Result::Exact(const Position& pos, int score) noexcept
{
	return Certain(pos.EmptyCount(), score);
}

Result Result::ExactFailHigh(const Position& pos, int score) noexcept
{
	return Result::CertainFailHigh(pos.EmptyCount(), score);
}

Result Result::ExactFailLow(const Position& pos, int score) noexcept
{
	return Result::CertainFailLow(pos.EmptyCount(), score);
}

Result Result::ExactFailHard(const Request& request, const Position& pos, int score) noexcept
{
	if (score > request)
		return Result::ExactFailHigh(pos, request.window.upper());
	if (score < request)
		return Result::ExactFailLow(pos, request.window.lower());
	return Result::Exact(pos, score);
}

Result Result::ExactFailSoft(const Request& request, const Position& pos, int score) noexcept
{
	if (score > request)
		return Result::ExactFailHigh(pos, score);
	if (score < request)
		return Result::ExactFailLow(pos, score);
	return Result::Exact(pos, score);
}

bool Result::operator>(const Request& request) const noexcept
{
	return (intensity >= request.intensity) && (window > request.window);
}