#include "Core/Core.h"
#include "Search/Search.h"
#include "IO/IO.h"
#include <vector>
#include <iostream>
#include <execution>

void TimePatternEval()
{
	unsigned int seed_1 = 4182;
	unsigned int seed_2 = 5899;
	int size = 100'000;
	int empty_count = 20;
	std::vector<Position> pos = PosGen::RandomPlayed{ seed_1, seed_2, empty_count }(size);

	PatternEval pattern_eval = DefaultPatternEval();

	auto start = std::chrono::high_resolution_clock::now();
	for (const Position& p : pos)
		pattern_eval.Eval(p);
	auto stop = std::chrono::high_resolution_clock::now();

	std::cout << "Avg time to eval a pattern: " << (stop - start) / size << std::endl << std::endl;
}

int main(int argc, char* argv[])
{
	TimePatternEval();

	int random_seed_1 = 4182;
	int random_seed_2 = 5899;
	int size = 100;

	std::vector<PuzzleProject> projects;
	for (int e = 1; e < 60; e++)
	{
		PuzzleProject proj;

		PosGen::RandomPlayed rnd(random_seed_1 + e, random_seed_2 + e, e);
		std::generate_n(
			std::back_inserter(proj),
			size,
			[&rnd]() { return Puzzle(rnd()); }
		);

		projects.push_back(std::move(proj));
	}

	std::cout.imbue(std::locale(""));

	PatternEval pattern_eval = DefaultPatternEval();
	HashTablePVS tt{ 1'000'000 };

	Process<PuzzleProject>(std::execution::par, projects,
		[&](Puzzle& p) { p.Solve(IDAB{ tt, pattern_eval }); },
		[](const PuzzleProject& proj) {
			std::chrono::duration<double> duration = Duration(proj);
			uint64 nodes = Nodes(proj);
			#pragma omp critical
			std::cout
				<< std::setw(8) << proj.front().pos.EmptyCount() << " | "
				<< short_time_format(duration / proj.size()) << "/pos | "
				<< std::setw(11) << std::size_t(nodes / duration.count()) << " N/s | "
				<< duration.count() << "\n";
		});

	std::cout
		<< " Empties |     TTS     |      Speed      | Duration \n"
		<< "---------+-------------+-----------------+----------\n";
	return 0;
}