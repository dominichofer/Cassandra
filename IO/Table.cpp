#include "Table.h"
#include <ranges>

void Table::AddColumn(std::string title, int width, std::string content_format, bool visible)
{
	cols.emplace_back(visible, width, std::move(title), " " + content_format + " ");
}

void Table::PrintSeparator() const
{
	fmt::print("{}\n", fmt::join(
		cols
		| std::ranges::views::filter([](const auto& col) { return col.visible; })
		| std::ranges::views::transform([](const auto& col) { return std::string(col.width + 2, '-'); })
		, "+")
	);
}

void Table::PrintHeader() const
{
	fmt::print("{}\n", fmt::join(
		cols
		| std::ranges::views::filter([](const auto& col) { return col.visible; })
		| std::ranges::views::transform([](const auto& col) { return fmt::format(" {:^{}} ", col.title, col.width); })
		, "|")
	);
	PrintSeparator();
}
