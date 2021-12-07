#include "Hashtable.h"
#include "Perft.h"
#include "Core/Core.h"
#include "IO/IO.h"
#include <numeric>
#include <chrono>
#include <string>

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
	std::size_t RAM = 1_GB;

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

	Table table;
	table.AddColumn("depth", 5, "{:>5}");
	table.AddColumn("Positions", 21, "{:>21L}");
	table.AddColumn("correct", 7, "{:^7}");
	table.AddColumn("Time [s]", 16, "{:>#16.3f}");
	table.AddColumn("Pos/s", 16, "{:>16.0Lf}");
	table.PrintHeader();

	for (int d = 1; d <= depth; d++)
	{
		engine->clear();
		auto start = std::chrono::high_resolution_clock::now();
		auto result = engine->calculate(d);
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1'000.0;
		bool correct = (Correct(d) == result);

		if (duration)
			table.PrintRow(d, result, correct, duration, result / duration);
		else
			table.PrintRow(d, result, correct, duration, Table::Empty);
	}
	return 0;
}