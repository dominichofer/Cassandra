#pragma once
#include <format>
#include <iostream>
#include <string>

class Table
{
protected:
	std::string title;
	std::string format;
public:
	Table(std::string title, std::string format) noexcept;

	void PrintHeader() const;
	void PrintSeparator() const;
	void PrintRow(auto... content) const
	{
		std::cout << std::vformat(format, std::make_format_args(content...)) << std::endl;
	}
};