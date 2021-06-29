#include "Objects.h"

//Result Result::CertainFailSoft(const Intensity& i, int depth, int score) noexcept
//{
//	if (score > i)
//		return Result::CertainFailHigh(depth, score);
//	if (score < i)
//		return Result::CertainFailLow(depth, score);
//	return Result::Certain(depth, score);
//}

//Result Result::Exact(const Position& pos, int score) noexcept
//{
//	return Certain(pos.EmptyCount(), score);
//}
//
//Result Result::ExactFailHigh(const Position& pos, int score) noexcept
//{
//	return Result::CertainFailHigh(pos.EmptyCount(), score);
//}
//
//Result Result::ExactFailLow(const Position& pos, int score) noexcept
//{
//	return Result::CertainFailLow(pos.EmptyCount(), score);
//}

//Result Result::ExactFailHard(const Request& request, const Position& pos, int score) noexcept
//{
//	if (score > request)
//		return Result::ExactFailHigh(pos, request.window.upper());
//	if (score < request)
//		return Result::ExactFailLow(pos, request.window.lower());
//	return Result::Exact(pos, score);
//}
//
//Result Result::ExactFailSoft(const Request& request, const Position& pos, int score) noexcept
//{
//	if (score > request)
//		return Result::ExactFailHigh(pos, score);
//	if (score < request)
//		return Result::ExactFailLow(pos, score);
//	return Result::Exact(pos, score);
//}
//
//bool Result::operator>(const Request& request) const noexcept
//{
//	return (intensity >= request.intensity) && (window > request.window);
//}