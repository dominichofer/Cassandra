#include "Core/Core.h"
#include "Search/Search.h"
#include "IO/IO.h"
#include <vector>
#include <iostream>
#include <execution>

using namespace std::chrono_literals;

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

void TimeLateEndgame()
{
	int random_seed_1 = 4182;
	int random_seed_2 = 5899;
	int size = 1'000'000;
	int e = 7;

	std::vector<Puzzle> puzzles;
	puzzles.reserve(size);
	PosGen::RandomPlayed posgen(random_seed_1 + e, random_seed_2 + e, e);
	for (const Position& pos : posgen(size))
		puzzles.push_back(Puzzle{ pos, Puzzle::Task{ Request::ExactScore(pos) } });

	auto ex = CreateExecutor<Puzzle>(std::execution::par, puzzles,
		[](Puzzle& p) { p.Solve(AlphaBetaFailSoft{}); });

	auto start = std::chrono::high_resolution_clock::now();
	ex->Execute();
	auto stop = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<double>(stop - start);
	auto nodes = Nodes(puzzles);
	std::cout << "e7 wall clock time:"
		<< short_time_format(duration / size) << "/pos, "
		<< std::size_t(nodes / duration.count()) << " N/s, "
		<< duration.count() << " s" << std::endl << std::endl;
}

int main(int argc, char* argv[])
{
	std::cout.imbue(std::locale(""));
	TimePatternEval();
	TimeLateEndgame();

	int random_seed_1 = 4182;
	int random_seed_2 = 5899;
	int size = 100;
	PatternEval pattern_eval = DefaultPatternEval();
	HashTablePVS tt{ 1'000'000 };

	std::cout
		<< " Empties |     TTS     |      Speed      | Duration \n"
		<< "---------+-------------+-----------------+----------\n";

	std::vector<std::vector<Puzzle>> groups;
	for (int e = 15; e < 30; e++)
	{
		std::vector<Puzzle> group;
		group.reserve(size);
		PosGen::RandomPlayed posgen(random_seed_1 + e, random_seed_2 + e, e);
		for (const Position& pos : posgen(size))
			group.push_back(Puzzle{ pos, Puzzle::Task{ Request::ExactScore(pos) } });
		groups.push_back(std::move(group));
	}

	auto ex = CreateExecutor<Puzzle>(std::execution::par, groups,
		[&](Puzzle& p) { p.Solve(IDAB{ tt, pattern_eval }); });

	Metronome reporter(1s, [&]() {
		static std::size_t index = 0;
		while (index < groups.size() and ex->IsDone(index)) {
			auto duration = Duration(groups[index]);
			auto nodes = Nodes(groups[index]);
			auto e = groups[index].front().pos.EmptyCount();
			std::cout
				<< std::setw(8) << e  << " | "
				<< short_time_format(duration / size) << "/pos | "
				<< std::setw(11) << std::size_t(nodes / duration.count()) << " N/s | "
				<< duration.count() << "\n";
			index++;
		}
	});

	reporter.Start();
	ex->Execute();
	reporter.Force();

	return 0;
}