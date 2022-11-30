#pragma once
#include "fmt/format.h"
#include "fmt/chrono.h"
//#include "fmt/ranges.h"

template <typename T>
struct fmt::formatter<std::optional<T>>
{
	std::string fmt;
	int length = 0;

	template<typename ParseContext>
	constexpr auto parse(ParseContext& ctx)
	{
		auto first_of = std::ranges::find_first_of(ctx, std::string_view{ "0123456789}" });
		std::from_chars(first_of, ctx.end(), length);

		auto fmt_end = std::ranges::find(ctx, '}');
		fmt = std::string{ ctx.begin(), fmt_end };
		return fmt_end;
	}

	template <typename FormatContext>
	auto format(std::optional<T> opt, FormatContext& ctx)
	{
		if (opt.has_value())
			return fmt::format_to(ctx.out(), "{:" + fmt + "}", opt.value());
		return fmt::format_to(ctx.out(), "{:{}}", "", length);
	}
};
