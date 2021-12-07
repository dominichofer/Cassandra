#include "Table.h"

void Table::AddColumn(std::string title, int width, std::string content_format, bool visible)
{
	cols.emplace_back(visible, width, std::move(title), " " + content_format + " ");
}

void Table::PrintSeparator()
{
	fmt::print("{}\n", fmt::join(
		cols
		| ranges::views::filter([](const auto& col) { return col.visible; })
		| ranges::views::transform([](const auto& col) { return std::string(col.width + 2, '-'); })
		, "+")
	);
}

void Table::PrintHeader()
{
	fmt::print("{}\n", fmt::join(
		cols
		| ranges::views::filter([](const auto& col) { return col.visible; })
		| ranges::views::transform([](const auto& col) { return fmt::format(" {:^{}} ", col.title, col.width); })
		, "|")
	);
	PrintSeparator();
}
