#pragma once
#include "Format.h"
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
		fmt::print(format + '\n', content...);
	}
};