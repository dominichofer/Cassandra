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
	PatternEval pattern_eval = DefaultPatternEval();
	HashTablePVS tt{ 1'000'000 };

	std::cout.imbue(std::locale(""));
	std::cout
		<< " Empties |     TTS     |      Speed      | Duration \n"
		<< "---------+-------------+-----------------+----------\n";

	std::vector<std::vector<Puzzle>> groups;
	for (int e = 0; e < 60; e++)
	{
		std::vector<Puzzle> group;
		group.reserve(size);
		PosGen::RandomPlayed posgen(random_seed_1 + e, random_seed_2 + e, e);
		for (const Position& pos : posgen(size))
			group.push_back(Puzzle{ pos, Puzzle::Task{ Request::ExactScore(pos) } });
		groups.push_back(std::move(group));
	}

	CreateExecutor(std::execution::par, groups)

	Metronome auto_saver(60s, [](const auto& ex) { ex.lock(); });

	for (int e = 0; e < 60; e++)
	{
		std::vector<Puzzle> puzzles;
		puzzles.reserve(size);
		PosGen::RandomPlayed posgen(random_seed_1 + e, random_seed_2 + e, e);
		for (const Position& pos : posgen(size))
			puzzles.push_back(Puzzle{ pos, Puzzle::Task{ Request::ExactScore(pos) } });

		Process(std::execution::par, puzzles,
			[&](Puzzle& p) { p.Solve(IDAB{ tt, pattern_eval }); });

		auto duration = Duration(puzzles);
		auto nodes = Nodes(puzzles);
		std::cout
			<< std::setw(8) << e << " | "
			<< short_time_format(duration / size) << "/pos | "
			<< std::setw(11) << std::size_t(nodes / duration.count()) << " N/s | "
			<< duration.count() << "\n";
	}
	return 0;
}