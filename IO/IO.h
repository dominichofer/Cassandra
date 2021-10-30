#pragma once
#include "Core/Core.h"
#include "Database.h"
#include "FForum.h"
#include "Integers.h"
#include "PosScore.h"
#include "File.h"
#include "PatternEval.h"
#include <string>

[[nodiscard]] Field ParseField(const std::string&);

[[nodiscard]] Position ParsePosition_SingleLine(const std::string&) noexcept(false);

[[nodiscard]] std::string short_time_format(std::chrono::duration<double> duration);

template <typename T>
std::string to_string(const std::vector<T>& vec)
{
	using std::to_string;

	if (vec.empty())
		return "()";

	std::string s = "(";
	for (std::size_t i = 0; i < vec.size() - 1; i++)
		s += to_string(vec[i]) + ", ";
	return s + to_string(vec.back()) + ")";
}