#pragma once
#include "fmt/format.h"
#include "fmt/chrono.h"
#include "fmt/ranges.h"
#include "Search/Search.h"

template <>
struct fmt::formatter<Request> : fmt::formatter<std::string>
{
	auto format(const Request& r, format_context& ctx)
	{
		if (r.HasMove())
			return fmt::formatter<std::string>::format(to_string(r.move) + " " + to_string(r.intensity), ctx);
		else
			return fmt::formatter<std::string>::format(to_string(r.intensity), ctx);
	}
};
