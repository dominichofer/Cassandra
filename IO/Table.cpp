#include "Table.h"
#include <ranges>

Table::Table(std::string title, std::string format) noexcept
	: title(std::move(title))
	, format(std::move(format))
{}

void Table::PrintHeader() const
{
	fmt::print("{}\n", title);
	PrintSeparator();
}

void Table::PrintSeparator() const
{
	fmt::print("{}\n", fmt::join(title | std::views::transform([](char c) { return c == '|' ? '+' : '-'; }), ""));
}
