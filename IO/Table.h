#pragma once
#include "Format.h"
#include <string>
#include <optional>
#include <vector>

namespace
{
	template <typename T>
	constexpr bool is_optional(T const&) { return false; }

	template <typename T>
	constexpr bool is_optional(std::optional<T> const&) { return true; }
}

class Table
{
	struct EmptyType {};

	struct Column
	{
		bool visible;
		int width;
		std::string title, format;
	};
	std::vector<Column> cols;

	void print_empty(int index) { fmt::print(std::string(cols[index].width + 2, ' ')); }
	void print_visible_content(int index, auto content) { fmt::print(cols[index].format, content); }
	void print_visible_content(int index, EmptyType) { print_empty(index); }
	void print_visible_content(int index, std::nullopt_t) { print_empty(index); }
	template <typename T>
	void print_visible_content(int index, std::optional<T> content)
	{
		if (content.has_value())
			print_visible_content(index, content.value());
		else
			print_empty(index);
	}
	void print_content(int index, auto content)
	{
		if (cols[index].visible)
		{
			print_visible_content(index, content);
			if (index < cols.size() - 1)
				fmt::print("|");
		}
	}
	void print_content(int index, auto content, auto ... rest)
	{
		print_content(index, content);
		print_content(index + 1, rest...);
	}
public:
	static inline EmptyType Empty{};

	void AddColumn(std::string title, int width, std::string content_format, bool visible = true);

	void PrintSeparator();
	void PrintHeader();

	void PrintRow(auto ... content)
	{
		print_content(0, content ...);
		fmt::print("\n");
	}
};