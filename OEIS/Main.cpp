#include "Core/Core.h"
#include "IO/IO.h"
#include <algorithm>
#include <execution>
#include <iostream>
#include <numeric>
#include <vector>

void PrintHelp()
{
	std::cout
		<< "   -d    Depth of perft.\n"
		<< "   -RAM  Number of hash table bytes.\n"
		<< "   -h    Prints this help."
		<< std::endl;
}

std::size_t NumberOfDifferentPositions(const std::vector<Position>& all)
{
	return std::inner_product(all.begin() + 1, all.end(),
							  all.begin(),
							  std::size_t(1), std::plus(), std::not_equal_to());
}

// Counts positions that occure exactly once in the input.
std::size_t NumberOfUniques(const std::vector<Position>& pos)
{
	const int64_t size = static_cast<int64_t>(pos.size());
	if (size < 2)
		return size;

	int64_t sum = (pos[0] != pos[1]) ? 1 : 0;
	#pragma omp parallel for reduction(+:sum)
	for (int64_t i = 1; i < size - 1; i++)
		if ((pos[i-1] != pos[i]) and (pos[i] != pos[i+1]))
			sum++;
	if (pos[size-2] != pos[size-1])
		sum++;
	return sum;
}

int main()
{
	std::locale::global(std::locale(""));

	Table table;
	table.AddColumn("Plies", 5, "{:>5}");
	table.AddColumn("A124005", 12, "{:>12L}");
	table.AddColumn("A124006", 12, "{:>12L}");
	table.AddColumn("A125528", 12, "{:>12L}");
	table.AddColumn("A125529", 12, "{:>12L}");
	table.AddColumn("Time [s]", 9, "{:>9.3f}");
	table.PrintHeader();

	for (int plies = 1; plies < 20; plies++)
	{
		auto start = std::chrono::high_resolution_clock::now();

		auto gen = Children(Position::Start(), plies, true);
		std::vector<Position> all(gen.begin(), gen.end());
		std::sort(std::execution::par, all.begin(), all.end());

		std::size_t num_different_pos = NumberOfDifferentPositions(all);
		std::size_t num_unique_pos = NumberOfUniques(all);

		std::transform(std::execution::par,
			all.begin(), all.end(),
			all.begin(),
			[](const Position& pos) { return Position(pos.Player() | pos.Opponent(), 0); });
		std::sort(std::execution::par, all.begin(), all.end());

		std::size_t num_different_shapes = NumberOfDifferentPositions(all);
		std::size_t num_unique_shapes = NumberOfUniques(all);

		auto stop = std::chrono::high_resolution_clock::now();


		table.PrintRow(plies,
			num_different_pos, num_unique_pos, 
			num_different_shapes, num_unique_shapes,
			std::chrono::round<std::chrono::milliseconds>(stop - start).count() / 1000.0);
	}
	return 0;
}

// Plies |   A124005    |   A124006    |   A125528    |   A125529    | Time [s]
//-------+--------------+--------------+--------------+--------------+-----------
//     1 |            4 |            4 |            4 |            4 |     0.001
//     2 |           12 |           12 |           12 |           12 |     0.000
//     3 |           54 |           52 |           54 |           52 |     0.000
//     4 |          236 |          228 |          220 |          196 |     0.000
//     5 |        1'288 |        1'192 |        1'130 |          932 |     0.000
//     6 |        7'092 |        6'160 |        5'568 |        3'944 |     0.001
//     7 |       42'614 |       33'344 |       26'966 |       14'020 |     0.004
//     8 |      269'352 |      191'380 |      132'037 |       53'556 |     0.029
//     9 |    1'743'560 |    1'072'232 |      589'652 |      165'584 |     0.193
//    10 |   11'922'442 |    6'416'600 |    2'601'811 |      507'656 |     1.447
//    11 |   80'209'268 |   35'990'544 |   10'147'378 |    1'251'972 |    12.869
//    12 |  562'280'115 |  212'278'256 |   38'356'054 |    2'988'984 |   126.412
