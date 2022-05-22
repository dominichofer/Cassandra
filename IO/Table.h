#pragma once
#include "Format.h"
#include <string>
#include <optional>
#include <vector>

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

	void print_empty(int index) const { fmt::print(std::string(cols[index].width + 2, ' ')); }
	void print_visible_content(int index, auto content) const { fmt::print(cols[index].format, content); }
	void print_visible_content(int index, EmptyType) const { print_empty(index); }
	void print_visible_content(int index, std::nullopt_t) const { print_empty(index); }
	template <typename T>
	void print_visible_content(int index, std::optional<T> content) const
	{
		if (content.has_value())
			print_visible_content(index, content.value());
		else
			print_empty(index);
	}

	void print_content(int index, auto content, auto ... rest) const
	{
		print_content(index, content);
		print_content(index + 1, rest...);
	}
public:
	void print_content(int index, auto content) const
	{
		if (cols[index].visible)
		{
			print_visible_content(index, content);
			if (index < cols.size() - 1)
				fmt::print("|");
		}
	}

	Table() { std::locale::global(std::locale("")); }
	static inline EmptyType Empty{};

	void AddColumn(std::string title, int width, std::string content_format, bool visible = true);

	void PrintSeparator() const;
	void PrintHeader() const;

	void PrintRow(auto ... content) const
	{
		print_content(0, content ...);
		fmt::print("\n");
	}
};