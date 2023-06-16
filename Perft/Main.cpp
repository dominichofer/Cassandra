#include "Hashtable.h"
#include "Perft.h"
#include "Core/Core.h"
#include "IO/IO.h"
#include <numeric>
#include <chrono>
#include <string>
#include <iostream>

void PrintHelp()
{
	std::cout
		<< "   -d    Depth of perft.\n"
		<< "   -RAM  Number of hash table bytes.\n"
		<< "   -h    Prints this help."
		<< std::endl;
}

int main(int argc, char* argv[])
{
	std::locale::global(std::locale(""));
	int depth = 16;
	std::size_t RAM = 4_GB;

	for (int i = 0; i < argc; i++)
	{
		if (std::string(argv[i]) == "-d") depth = std::stoi(argv[++i]);
		else if (std::string(argv[i]) == "-RAM") RAM = ParseBytes(argv[++i]);
		else if (std::string(argv[i]) == "-h") { PrintHelp(); return 0; }
	}

	std::unique_ptr<BasicPerft> engine;
	if (RAM)
		engine = std::make_unique<HashTablePerft>(RAM, 6);
	else
		engine = std::make_unique<UnrolledPerft>(6);

	Table table{
		"depth|      Positions      |correct|    Time [s]    |     Pos/s      ",
		"{:>5}|{:>21L}|{:^7}|{:>#16.3f}|{:>16L}"
	};
	table.PrintHeader();

	for (int d = 4; d <= depth; d++)
	{
		engine->clear();
		auto start = std::chrono::high_resolution_clock::now();
		auto result = engine->calculate(d);
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1'000.0;
		bool correct = (Correct(d) == result);

		table.PrintRow(d, result, correct, duration, (uint64_t)(duration ? result / duration : 0));
	}
	return 0;
}