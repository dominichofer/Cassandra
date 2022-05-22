#pragma once
#include "Format.h"
#include <string>

template <typename T>
std::string to_string(T&& t)
#ifndef __NVCC__
requires requires (T&& t) { t.to_string(); }
#endif
{
	return t.to_string();
}

template <typename T>
struct to_string_formatter : fmt::formatter<std::string>
{
	auto format(const T& t, fmt::format_context& ctx)
	{
		return fmt::formatter<std::string>::format(to_string(t), ctx);
	}
};

template <typename T>
#ifndef __NVCC__
requires requires (const T& t) { t.to_string(); }
#endif
struct fmt::formatter<T> : to_string_formatter<T>
{};


// TODO: Move this!
struct HalfOpenInterval
{
	int lower, upper;
};