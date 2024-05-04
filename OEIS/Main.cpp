#include "Board/Board.h"
#include "Search/Search.h"
#include <algorithm>
#include <execution>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <vector>

// std::set(pos).size()
std::size_t NumberOfDifferentPositions(const std::vector<Position>& pos)
{
	if (pos.empty())
		return 0;

	int64_t size = static_cast<int64_t>(pos.size());
	std::size_t sum = 1;
	#pragma omp parallel for reduction(+:sum)
	for (int64_t i = 1; i < size; i++)
		if (pos[i - 1] != pos[i])
			sum++;
	return sum;
}

// Counts positions that occure exactly once.
std::size_t NumberOfUniques(const std::vector<Position>& pos)
{
	int64_t size = static_cast<int64_t>(pos.size());
	if (size < 2)
		return size;

	std::size_t sum = (pos[0] != pos[1]) ? 1 : 0;
	#pragma omp parallel for reduction(+:sum)
	for (int64_t i = 1; i < size - 1; i++)
		if ((pos[i-1] != pos[i]) and (pos[i] != pos[i+1]))
			sum++;
	if (pos[size-2] != pos[size-1])
		sum++;
	return sum;
}

void PrintHelp()
{
	std::cout
		<< "A124004: Number of possible Reversi games at the end of the n-th ply. (https://oeis.org/A124004)\n"
		<< "A124005: Number of different Othello positions at the end of the n-th ply. (https://oeis.org/A124005)\n"
		<< "A124006: Number of Othello positions with unique realization at the end of the n-th ply. (https://oeis.org/A124006)\n"
		<< "A125528: Number of different Othello shapes at the end of the n-th ply. (https://oeis.org/A125528)\n"
		<< "A125529: Number of Othello shapes with unique realization at the end of the n-th ply. (https://oeis.org/A125529)\n"
		<< "   -d    Depth of perft.\n"
		<< "   -RAM  Number of hash table bytes.\n"
		<< "   -h    Prints this help."
		<< std::endl;
}

int main()
{
	std::locale::global(std::locale(""));

	Table table{
		"Plies|    A124004    |   A124005   |   A124006   |   A125528   |   A125529   | Time [s]",
		"{:>5}|{:>15L}|{:>13L}|{:>13L}|{:>13L}|{:>13L}| {}"
	};
	table.PrintHeader();

	std::vector<Position> all;
	all.reserve(4'000'000'000);

	const Position tombstone{ 0, 0 };
	const Position pos_1 = Play(Position::Start(), PossibleMoves(Position::Start())[0]); // causes symmetries = 4
	const std::size_t symmetries = 4;

	for (int plies = 1; plies < 20; plies++)
	{
		std::size_t games = 0;
		std::size_t num_different_pos = 0;
		std::size_t num_unique_pos = 0;
		std::size_t num_different_shapes = 0;
		std::size_t num_unique_shapes = 0;

		int part_count = 1;
		if (plies == 13) part_count = 2;
		if (plies == 14) part_count = 2 * 8;
		if (plies == 15) part_count = 2 * 8 * 8;

		auto start = std::chrono::high_resolution_clock::now();
		for (int part = 0; part < part_count; part++)
		{
			all.clear();
			for (Position pos : Children(pos_1, plies, true))
			{
				if (Hash(pos) % part_count == part)
				{
					games++;
					all.push_back(pos);
					if (all.size() == all.capacity()) // compress
					{
						std::sort(std::execution::par, all.begin(), all.end());
						for (uint64_t i = 1; i < all.size() - 1; i++)
							if (all[i - 1] == all[i] and all[i] == all[i + 1])
								all[i - 1] = tombstone;
						std::erase(all, tombstone);
					}
				}
			}
			std::sort(std::execution::par, all.begin(), all.end());

			num_different_pos += NumberOfDifferentPositions(all);
			num_unique_pos += NumberOfUniques(all);

			std::transform(std::execution::par,
				all.begin(), all.end(),
				all.begin(),
				[](const Position& pos) { return Position(pos.Player() | pos.Opponent(), 0); });
			std::sort(std::execution::par, all.begin(), all.end());

			num_different_shapes += NumberOfDifferentPositions(all);
			num_unique_shapes += NumberOfUniques(all);
		}

		auto stop = std::chrono::high_resolution_clock::now();

		table.PrintRow(plies,
			symmetries * games,
			symmetries * num_different_pos,
			symmetries * num_unique_pos,
			symmetries * num_different_shapes,
			symmetries * num_unique_shapes,
			HH_MM_SS(stop - start));
	}
	return 0;
}

// Plies |    A124005    |    A124006    |    A125528    |    A125529    | Time [s]
//-------+---------------+---------------+---------------+---------------+-----------
//     1 |             4 |             4 |             4 |             4 |     0.001
//     2 |            12 |            12 |            12 |            12 |     0.000
//     3 |            54 |            52 |            54 |            52 |     0.000
//     4 |           236 |           228 |           220 |           196 |     0.000
//     5 |         1'288 |         1'192 |         1'130 |           932 |     0.000
//     6 |         7'092 |         6'160 |         5'568 |         3'944 |     0.001
//     7 |        42'614 |        33'344 |        26'966 |        14'020 |     0.004
//     8 |       269'352 |       191'380 |       132'037 |        53'556 |     0.026
//     9 |     1'743'560 |     1'072'232 |       589'652 |       165'584 |     0.179
//    10 |    11'922'442 |     6'416'600 |     2'601'811 |       507'656 |     1.396
//    11 |    80'209'268 |    35'990'544 |    10'147'378 |     1'251'972 |    12.699
//    12 |   562'280'115 |   212'278'256 |    38'356'054 |     2'988'984 |   126.574
//    13 | 3'772'081'046 | 1'145'811'772 |   129'698'048 |     5'904'880 |  9958.995
