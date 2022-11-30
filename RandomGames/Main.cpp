#include "Core/Core.h"
#include <iostream>
#include <random>
#include <set>
#include <vector>

void PrintHelp()
{
	std::cout
		<< "Outputs 'n' randomly played games starting from unique positions with 'e' empty fields.\n"
		<< "   -e <int>   Number of empty fields.\n"
		<< "   -n <int>   Number of games to play.\n"
		<< "   -h         Prints this help."
		<< std::endl;
}

int main(int argc, char* argv[])
{
	int empty_count = 50;
	int num = 0;
	for (int i = 0; i < argc; i++)
	{
		if (std::string(argv[i]) == "-e") empty_count = std::stoi(argv[++i]);
		else if (std::string(argv[i]) == "-n") num = std::stoi(argv[++i]);
		else if (std::string(argv[i]) == "-h") { PrintHelp(); return 0; }
	}

	std::mt19937_64 rnd_engine(std::random_device{}());

	std::vector<Position> samples;
	std::ranges::sample(
		UniqueChildren(Position::Start(), empty_count),
		std::back_inserter(samples),
		num, rnd_engine);

	for (const Game& game : RandomGamesFrom(samples))
		std::cout << to_string(game) << '\n';
	return 0;
}
