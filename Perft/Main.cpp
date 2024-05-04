#include "Base/Base.h"
#include "Hashtable.h"
#include "Perft.h"
#include <chrono>
#include <iostream>
#include <numeric>
#include <string>

void PrintHelp()
{
	std::cout
		<< "   -d     Depth of perft.\n"
		<< "   -tt    Number of hash table bits.\n"
		<< "   -cuda  Use CUDA.\n"
		<< "   -h     Prints this help."
		<< std::endl;
}

int main(int argc, char* argv[])
{
	std::locale::global(std::locale(""));
	int depth = 20;
	std::size_t tt_size = 1'000'000'000;
	bool cuda = false;

	for (int i = 0; i < argc; i++)
	{
		std::string_view arg{ argv[i] };
		if (arg == "-d") depth = std::stoi(argv[++i]);
		else if (arg == "-tt") tt_size = std::stoi(argv[++i]);
		else if (arg == "-cuda") cuda = true;
		else if (arg == "-h") { PrintHelp(); return 0; }
	}

	Table table{
		"depth|       Positions      |correct|    Time [s]    |      Pos/s      ",
		"{:>5}|{:>22L}|{:^7}|{:>16}|{:>17L}"
	};
	table.PrintHeader();

	HashTable tt{ tt_size };
	Perft engine{ tt, cuda };
	for (int d = 10; d <= depth; d++)
	{
		tt.Clear();
		auto start = std::chrono::high_resolution_clock::now();
		auto result = engine.calculate(d);
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1'000.0;
		bool correct = (Correct(d) == result);

		table.PrintRow(d, result, correct, HH_MM_SS(stop - start), static_cast<uint64_t>(duration ? result / duration : 0));
	}
	return 0;
}