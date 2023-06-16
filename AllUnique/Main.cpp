#include "Core/Core.h"
#include "IO/IO.h"
#include <iostream>

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
	std::vector<PosScore> ps = LoadPosScoreFile("..\\data\\endgame.ps");
	for (auto x : ps)
		std::cout << std::to_string(x.score) << std::endl;
	//int empty_count = 60;
	//for (int i = 0; i < argc; i++)
	//{
	//	std::string arg = argv[i];
	//	if (arg == "-e") empty_count = std::stoi(argv[++i]);
	//	if (arg == "-h") { PrintHelp(); return 0; }
	//}

	//for (Position pos : UniqueChildren(Position::Start(), empty_count))
	//	std::cout << to_string(pos) << '\n';
	return 0;
}