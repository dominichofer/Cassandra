#include "Core/Core.h"
#include <iostream>
#include <set>

void PrintHelp()
{
	std::cout
		<< "Outputs all unique children with 'e' empty fields.\n"
		<< "   -e <int>        Number of empty fields.\n"
		<< "   -h              Prints this help."
		<< std::endl;
}

int main(int argc, char* argv[])
{
	int empty_count = 60;
	for (int i = 0; i < argc; i++)
	{
		if (std::string(argv[i]) == "-e") empty_count = std::stoi(argv[++i]);
		else if (std::string(argv[i]) == "-h") { PrintHelp(); return 0; }
	}

	for (Position pos : UniqueChildren(Position::Start(), empty_count))
		std::cout << to_string(pos) << '\n';
	return 0;
}