#include "Engine/NegaMaxSearch.h"
#include "Engine/AlphaBetaFailHardSearch.h"
#include "Engine/AlphaBetaFailSoftSearch.h"
#include "Engine/PVSearch.h"
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

class Puzzles
{
	std::vector<Puzzle> puzzles;
public:
	Puzzles(std::vector<Puzzle> puzzles) : puzzles(std::move(puzzles)) {}

	Puzzle& operator[](std::size_t i) noexcept { return puzzles[i % puzzles.size()]; }
	std::size_t size() noexcept { return puzzles.size(); }
	auto begin() noexcept { return puzzles.begin(); }
	auto end() noexcept { return puzzles.end(); }
};

class PuzzleLibrary
{
	std::vector<Puzzles> library;
public:
	PuzzleLibrary(std::size_t max_empty_count)
	{
		const std::size_t positions_per_empty_count = 100'000;
		const std::size_t seed = 65481265;

		for (std::size_t empty_count = 0; empty_count <= max_empty_count; empty_count++)
		{
			PosGen::Random_with_empty_count rnd(empty_count, seed);
			std::vector<Puzzle> puzzles;
			puzzles.reserve(positions_per_empty_count);
			std::generate_n(
				std::back_inserter(puzzles),
				positions_per_empty_count,
				[&rnd, empty_count]() { return Puzzle::Exact(rnd()); }
			);
			library.emplace_back(puzzles);
		}
	}

	Puzzles& operator[](std::size_t empty_count)
	{
		return library[empty_count];
	}
};

void print(std::chrono::nanoseconds time, std::size_t empty_count, std::size_t puzzles, std::size_t node_count)
{
	const auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(time);
	const auto duration_per_pos = duration / puzzles;
	const auto duration_per_node = duration / node_count;

	std::wcout.imbue(std::locale(""));
	std::wcout
		<< empty_count << L" Empties: "
		<< short_time_format(duration_per_pos) << L"/pos, "
		<< std::setprecision(2) << std::scientific << duration_per_pos.count() << " s/pos, "
		<< std::size_t(1.0 / duration_per_pos.count()) << " pos/s, "
		<< std::size_t(1.0 / duration_per_node.count()) << " N/s" << std::endl;
}

template <typename Algorithm>
void Benchmark_serial(PuzzleLibrary& library, const std::size_t empty_count, std::size_t sample_size)
{
	auto puzzles = library[empty_count];
	std::size_t node_count = 0;

	auto start = std::chrono::high_resolution_clock::now();
	//#pragma omp parallel for reduction(+:node_count)
	for (int64_t i = 0; i < sample_size; i++)
	{
		Algorithm algorithm;
		puzzles[i].Solve(algorithm);
		node_count += puzzles[i].Result().value().node_count;
	}
	auto stop = std::chrono::high_resolution_clock::now();

	print(stop - start, empty_count, sample_size, node_count);
}
template <>
void Benchmark_serial<Search::PVSearch>(PuzzleLibrary& library, const std::size_t empty_count, std::size_t sample_size)
{
	auto puzzles = library[empty_count];
	std::size_t node_count = 0;
	HashTablePVS tt{ 1'000'000 };

	auto start = std::chrono::high_resolution_clock::now();
	//#pragma omp parallel for reduction(+:node_count)
	for (int64_t i = 0; i < sample_size; i++)
	{
		Search::PVSearch algorithm{ tt };
		puzzles[i].Solve(algorithm);
		node_count += puzzles[i].Result().value().node_count;
	}
	auto stop = std::chrono::high_resolution_clock::now();

	print(stop - start, empty_count, sample_size, node_count);
}

int main(int argc, char* argv[])
{
	std::wcout << L"Creating data...";
	PuzzleLibrary library(20);
	std::wcout << L" done." << std::endl;

	//std::cout << "NegaMax serial\n";
	//Benchmark_serial<Search::NegaMax>(library, 0, 50'000'000);
	//Benchmark_serial<Search::NegaMax>(library, 1, 50'000'000);
	//Benchmark_serial<Search::NegaMax>(library, 2, 20'000'000);
	//Benchmark_serial<Search::NegaMax>(library, 3,  5'000'000);
	//Benchmark_serial<Search::NegaMax>(library, 4,  2'000'000);
	//Benchmark_serial<Search::NegaMax>(library, 5,    500'000);
	//Benchmark_serial<Search::NegaMax>(library, 6,    100'000);
	//Benchmark_serial<Search::NegaMax>(library, 7, 	  20'000);
	//Benchmark_serial<Search::NegaMax>(library, 8, 	   3'000);
	//Benchmark_serial<Search::NegaMax>(library, 9, 	     500);
	//Benchmark_serial<Search::NegaMax>(library, 10,       100);

	//std::cout << "\nAlphaBetaFailHard serial\n";
	//Benchmark_serial<Search::AlphaBetaFailHard>(library, 0, 50'000'000);
	//Benchmark_serial<Search::AlphaBetaFailHard>(library, 1, 50'000'000);
	//Benchmark_serial<Search::AlphaBetaFailHard>(library, 2, 20'000'000);
	//Benchmark_serial<Search::AlphaBetaFailHard>(library, 3,  5'000'000);
	//Benchmark_serial<Search::AlphaBetaFailHard>(library, 4,  2'000'000);
	//Benchmark_serial<Search::AlphaBetaFailHard>(library, 5,    500'000);
	//Benchmark_serial<Search::AlphaBetaFailHard>(library, 6,    200'000);
	//Benchmark_serial<Search::AlphaBetaFailHard>(library, 7,    100'000);
	//Benchmark_serial<Search::AlphaBetaFailHard>(library, 8, 	20'000);
	//Benchmark_serial<Search::AlphaBetaFailHard>(library, 9, 	 5'000);
	//Benchmark_serial<Search::AlphaBetaFailHard>(library, 10,     2'000);
	//Benchmark_serial<Search::AlphaBetaFailHard>(library, 11,       500);
	//Benchmark_serial<Search::AlphaBetaFailHard>(library, 12,       100);

	//std::cout << "\nAlphaBetaFailSoft serial\n";
	//Benchmark_serial<Search::AlphaBetaFailSoft>(library, 0, 50'000'000);
	//Benchmark_serial<Search::AlphaBetaFailSoft>(library, 1, 50'000'000);
	//Benchmark_serial<Search::AlphaBetaFailSoft>(library, 2, 20'000'000);
	//Benchmark_serial<Search::AlphaBetaFailSoft>(library, 3,  5'000'000);
	//Benchmark_serial<Search::AlphaBetaFailSoft>(library, 4,  2'000'000);
	//Benchmark_serial<Search::AlphaBetaFailSoft>(library, 5,    500'000);
	//Benchmark_serial<Search::AlphaBetaFailSoft>(library, 6,    200'000);
	//Benchmark_serial<Search::AlphaBetaFailSoft>(library, 7,    100'000);
	//Benchmark_serial<Search::AlphaBetaFailSoft>(library, 8, 	20'000);
	//Benchmark_serial<Search::AlphaBetaFailSoft>(library, 9, 	 5'000);
	//Benchmark_serial<Search::AlphaBetaFailSoft>(library, 10,     2'000);
	//Benchmark_serial<Search::AlphaBetaFailSoft>(library, 11,       500);
	//Benchmark_serial<Search::AlphaBetaFailSoft>(library, 12,       100);

	std::cout << "\nPVSearch serial\n";
	Benchmark_serial<Search::PVSearch>(library, 0, 50'000'000);
	Benchmark_serial<Search::PVSearch>(library, 1, 50'000'000);
	Benchmark_serial<Search::PVSearch>(library, 2, 20'000'000);
	Benchmark_serial<Search::PVSearch>(library, 3,  5'000'000);
	Benchmark_serial<Search::PVSearch>(library, 4,  2'000'000);
	Benchmark_serial<Search::PVSearch>(library, 5,  1'000'000);
	Benchmark_serial<Search::PVSearch>(library, 6,    200'000);
	Benchmark_serial<Search::PVSearch>(library, 7,    100'000);
	Benchmark_serial<Search::PVSearch>(library, 8, 	   20'000);
	Benchmark_serial<Search::PVSearch>(library, 9, 	    5'000);
	Benchmark_serial<Search::PVSearch>(library, 10,     2'000);
	Benchmark_serial<Search::PVSearch>(library, 11,       500);
	Benchmark_serial<Search::PVSearch>(library, 12,       100);
	Benchmark_serial<Search::PVSearch>(library, 13,       100);
	Benchmark_serial<Search::PVSearch>(library, 14,       100);
	Benchmark_serial<Search::PVSearch>(library, 15,       100);
	Benchmark_serial<Search::PVSearch>(library, 16,       100);
	Benchmark_serial<Search::PVSearch>(library, 17,       100);
	Benchmark_serial<Search::PVSearch>(library, 18,       100);
	Benchmark_serial<Search::PVSearch>(library, 19,       100);
	Benchmark_serial<Search::PVSearch>(library, 20,       100);

	return 0;
}