#include "Engine/NegaMaxSearch.h"
#include "Engine/AlphaBetaFailHardSearch.h"
#include "Engine/AlphaBetaFailSoftSearch.h"
#include "Core/Puzzle.h"
#include "Core/PositionGenerator.h"
#include "Core/Position.h"
#include "IO/Main.h"

#include <atomic>
#include <numeric>
#include <vector>
#include <iostream>
#include <omp.h>

using namespace std::chrono_literals;

std::vector<Puzzle> CreatePuzzles(std::size_t empty_count)
{
	std::vector<std::size_t> positions_per_empty_count =
	{
		1'000'000, // 0
		1'000'000,
		1'000'000,
		1'000'000,
		1'000'000,
		1'000'000, // 5
		1'000'000,
		  500'000,
		  250'000,
		  100'000,
		   50'000, // 10
		   25'000,
		   10'000,
			5'000,
			2'500,
			1'000, // 15
			  500,
			  250,
			  100,
			   50,
			   25, // 20
			   10,
			   10,
			   10,
			   10,
			   10, // 25
			   10,
			   10,
			   10,
			   10,
			   10  // 30
	};

	const std::size_t SEED = 65481265;
	PositionGenerator pos_gen(SEED);

	std::vector<Puzzle> ret;
	std::generate_n(std::back_inserter(ret), positions_per_empty_count[empty_count], 
		[&]() { 
			auto pos = pos_gen.Random(empty_count);
			return Puzzle(pos, Search::Intensity::Exact(pos));
		});
	return ret;
}

int main(int argc, char* argv[])
{
	for (std::size_t empty_count = 0; empty_count <= 30; empty_count++)
	{
		auto puzzles = CreatePuzzles(empty_count);

		auto start_time = std::chrono::high_resolution_clock::now();
		#pragma omp parallel for schedule(dynamic,1)
		for (int64_t i = 0; i < puzzles.size(); i++)
		{
			Search::AlphaBetaFailSoft algorithm;
			puzzles[i].Solve(algorithm);
		}
		auto end_time = std::chrono::high_resolution_clock::now();

		const std::size_t node_count = std::accumulate(puzzles.begin(), puzzles.end(), std::size_t{ 0 },
			[](std::size_t sum, const Puzzle& puzzle) { return sum + puzzle.Result().node_count; });

		const auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
		const auto duration_per_pos = duration / puzzles.size();
		const auto duration_per_node = duration / node_count;

		std::wcout.imbue(std::locale(""));
		std::wcout
			<< empty_count << L" Empties: "
			<< short_time_format(duration_per_pos) << L"/pos, "
			<< std::setprecision(2) << std::scientific << duration_per_pos.count() << " s/pos, "
			<< std::size_t(1.0 / duration_per_pos.count()) << " pos/s, "
			<< std::size_t(1.0 / duration_per_node.count()) << " N/s" << std::endl;
	}
	return 0;
}