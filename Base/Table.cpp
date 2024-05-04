#include "Table.h"
#include <iostream>

Table::Table(std::string title, std::string format) noexcept
	: title(std::move(title))
	, format(std::move(format))
{}

void Table::PrintHeader() const
{
	std::cout << title << '\n';
	PrintSeparator();
}

void Table::PrintSeparator() const
{
	for (char c : title)
		if (c == '|')
			std::cout << '+';
		else
			std::cout << '-';
	std::cout << '\n';
}
