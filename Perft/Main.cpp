#include "Hashtable.h"
#include "Perft.h"
#include "Core/Core.h"
#include "IO/IO.h"
#include <numeric>
#include <functional>
#include <execution>
#include <chrono>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>

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
	int depth = 16;
	std::size_t RAM = 1_GB;

	for (int i = 0; i < argc; i++)
	{
		if (std::string(argv[i]) == "-d") depth = std::stoi(argv[++i]);
		else if (std::string(argv[i]) == "-RAM") RAM = ParseBytes(argv[++i]);
		else if (std::string(argv[i]) == "-h") { PrintHelp(); return 0; }
	}

	std::unique_ptr<UnrolledPerft> engine;
	if (RAM)
		engine = std::make_unique<HashTablePerft>(RAM, 6);
	else
		engine = std::make_unique<UnrolledPerft>(6);

	std::cout << "depth|       Positions       |correct|       Time       |       Pos/s      \n";
	std::cout << "-----+-----------------------+-------+------------------+------------------\n";

	std::cout.imbue(std::locale(""));
	std::cout << std::setfill(' ') << std::boolalpha;

	for (int d = 1; d <= depth; d++)
	{
		engine->clear();
		const auto start = std::chrono::high_resolution_clock::now();
		const auto result = engine->calculate(d);
		const auto end = std::chrono::high_resolution_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1'000.0;

		std::cout << std::setw(4) << d << " |";
		std::cout << std::setw(22) << result << " |";
		std::cout << std::setw(6) << (Correct(d) == result) << " |";
		std::cout << std::setw(17) << duration << " |";
		if (duration)
			std::cout << std::setw(17) << int64(result / duration);
		std::cout << '\n';
	}
	return 0;
}