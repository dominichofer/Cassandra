#pragma once
#include "fmt/format.h"
#include "fmt/chrono.h"
//#include "fmt/ranges.h"

//template <typename T>
//requires requires (const T& t) { std::to_string(t); }
//struct fmt::formatter<T> : fmt::formatter<std::string>
//{
//	auto format(const T& t, format_context& ctx)
//	{
//		return fmt::formatter<std::string>::format(std::to_string(t), ctx);
//	}
//};

//template <typename T>
//requires requires (const T& t) { t.to_string(); }
//struct fmt::formatter<T> : fmt::formatter<std::string>
//{
//	auto format(const T& t, format_context& ctx)
//	{
//		return fmt::formatter<std::string>::format(t.to_string(), ctx);
//	}
//};

//template <typename T>
//concept HasToString = requires (const T& t) { to_string(t); };
//
//template <HasToString T>
//struct fmt::formatter<T> : fmt::formatter<std::string>
//{
//	auto format(const T& t, format_context& ctx)
//	{
//		return fmt::formatter<std::string>::format(to_string(t), ctx);
//	}
//};

//template <typename T>
//requires requires (const T& t) { to_string(t); }
//struct fmt::formatter<T> : fmt::formatter<std::string>
//{
//	auto format(const T& t, format_context& ctx)
//	{
//		return fmt::formatter<std::string>::format(to_string(t), ctx);
//	}
//};