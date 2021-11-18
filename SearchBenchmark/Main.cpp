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
	std::set<Position> set = generate_n_unique(size, PosGen::RandomlyPlayed{ seed_1, seed_2, empty_count });
	std::vector<Position> pos(set.begin(), set.end());

	AAGLEM pattern_eval = DefaultPatternEval();

	auto start = std::chrono::high_resolution_clock::now();
	for (const Position& p : pos)
		pattern_eval.Eval(p);
	auto stop = std::chrono::high_resolution_clock::now();

	std::cout << "Avg time to eval default patterns: " << (stop - start) / size << std::endl << std::endl;
}

void TimeLateEndgame() // AlphaBetaFailSoft
{
	int random_seed_1 = 4182;
	int random_seed_2 = 5899;
	int size = 1'000'000;
	int e = 7;

	std::vector<Puzzle> puzzles;
	puzzles.reserve(size);
	for (const Position& pos : generate_n_unique(size, PosGen::RandomlyPlayed(random_seed_1 + e, random_seed_2 + e, e)))
		puzzles.emplace_back(pos, Request::ExactScore(pos));

	auto ex = CreateExecutor(std::execution::par, puzzles,
		[](Puzzle& p) { p.Solve(AlphaBetaFailSoft{}); });

	auto start = std::chrono::high_resolution_clock::now();
	ex->Process();
	auto stop = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<double>(stop - start);
	auto nodes = Nodes(puzzles);
	std::cout << "e7 wall clock time:"
		<< short_time_format(duration / size) << "/pos, "
		<< std::size_t(nodes / duration.count()) << " N/s, "
		<< duration.count() << " s" << std::endl << std::endl;
}

template <typename ExecutionPolicy, PuzzleRange Range>
void Benchmark(ExecutionPolicy&& expo, auto E, auto d, Range&& puzzles)
{
	AAGLEM pattern_eval = DefaultPatternEval();
	HashTablePVS tt{ 10'000'000 };

	auto start = std::chrono::high_resolution_clock::now();
	Process(expo, puzzles, [&](Puzzle& p) { p.Solve(IDAB{ tt, pattern_eval }); });
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<double>(stop - start);

	auto size = std::ranges::distance(puzzles);
	auto nodes = Nodes(puzzles);
	std::cout
		<< std::format(std::locale(""), "{:8} | {:6} | {}/pos | {:13L} N/s | {:15L} | {:0.2f} s\n",
			E, d, short_time_format(duration / size), std::size_t(nodes / duration.count()), nodes, duration.count());
}

int main(int argc, char* argv[])
{
	DataBase<Puzzle> benchmark;
	for (int i = 0; i < 10; i++)
		for (int j = 0; j < 10; j++)
		{
			try
			{
				benchmark.Add(std::format(R"(G:\Reversi\play{}{}_benchmark.puz)", i, j));
			}
			catch (...) {}
		}

	//std::cout
	//	<< "Midgame\n"
	//	<< " Empties |  depth |     TTS     |       Speed       |       Nodes     | Duration \n"
	//	<< "---------+--------+-------------+-------------------+-----------------+----------\n";
	//for (int d = 0; d <= 10; d++)
	//{
	//	Request request(d);
	//	std::vector<Puzzle> to_bench;
	//	for (int i = 0; i < benchmark.size(); i++)
	//	{
	//		Puzzle& p = benchmark[i];
	//		if (p.pos.EmptyCount() > d and i % (d * d + 1) == 0)
	//		{
	//			to_bench.push_back(p);
	//			to_bench.back().clear();
	//			to_bench.back().push_back(request);
	//		}
	//	}
	//	Benchmark(std::execution::par, "", request, to_bench);
	//}

	//std::cout
	//	<< "\nSelective endgame\n"
	//	<< " Empties |  depth |     TTS     |       Speed       |       Nodes     | Duration \n"
	//	<< "---------+--------+-------------+-------------------+-----------------+----------\n";
	//for (int E = 0; E <= 25; E++)
	//{
	//	Request request(E, 1.1_sigmas);
	//	std::vector<Puzzle> to_bench;
	//	for (int i = 0; i < benchmark.size(); i++)
	//	{
	//		Puzzle& p = benchmark[i];
	//		if (p.pos.EmptyCount() == E and i % (E + 1) == 0)
	//		{
	//			to_bench.push_back(p);
	//			to_bench.back().clear();
	//			to_bench.back().push_back(request);
	//		}
	//	}
	//	Benchmark(std::execution::par, E, request, to_bench);
	//}

	std::cout
		<< "\nEndgame\n"
		<< " Empties |  depth |     TTS     |       Speed       |       Nodes     | Duration \n"
		<< "---------+--------+-------------+-------------------+-----------------+----------\n";
	for (int E = 0; E <= 25; E++)
	{
		Request request(E);
		std::vector<Puzzle> to_bench;
		for (int i = 0; i < benchmark.size(); i++)
		{
			Puzzle& p = benchmark[i];
			if (p.pos.EmptyCount() == E and i % (E * E / 10 + 1) == 0)
			{
				to_bench.push_back(p);
				to_bench.back().clear();
				to_bench.back().insert(request);
			}
		}
		Benchmark(std::execution::par, E, request, to_bench);
	}

	return 0;

	/*std::cout.imbue(std::locale(""));
	TimePatternEval();
	TimeLateEndgame();

	int random_seed_1 = 4182;
	int random_seed_2 = 5899;
	int size = 100;
	AAGLEM pattern_eval = DefaultPatternEval();
	HashTablePVS tt{ 1'000'000 };

	std::cout
		<< " Empties | depth |     TTS     |      Speed      |      Nodes     | Duration \n"
		<< "---------+-------+-------------+-----------------+----------------+----------\n";

	std::vector<std::vector<Puzzle>> groups;

	std::vector<int> depth_sizes = { 10'000, 10'000, 1'000, 1'000, 1'000, 100, 1'000, 100, 100, 100, 100 };
	for (int d = 0; d <= 10; d++)
	{
		std::vector<Puzzle> group;
		group.reserve(depth_sizes[d] * 31);
		for (int e = 20; e <= 50; e += 10)
		{
			for (const Position& pos : generate_n_unique(PosGen::RandomlyPlayed(random_seed_1 + d, random_seed_2 + d, e), depth_sizes[d]))
				group.push_back(Puzzle{ pos, Puzzle::Task{ Request(d) } });
		}
		groups.push_back(std::move(group));
	}
	for (int e = 15; e <= 30; e++)
	{
		std::vector<Puzzle> group;
		group.reserve(size);
		for (const Position& pos : generate_n_unique(PosGen::RandomlyPlayed(random_seed_1 + e, random_seed_2 + e, e), size))
			group.push_back(Puzzle{ pos, Puzzle::Task{ Request::ExactScore(pos) } });
		groups.push_back(std::move(group));
	}

	auto ex = CreateExecutor(std::execution::seq, groups,
		[&](Puzzle& p) { p.Solve(IDAB{ tt, pattern_eval }); });

	Metronome reporter(1s, [&]() {
		static std::size_t index = 0;
		while (index < groups.size() and ex->IsDone(index)) {
			auto duration = Duration(groups[index]);
			auto nodes = Nodes(groups[index]);
			auto e = groups[index].front().pos.EmptyCount();
			auto d = groups[index].front().tasks.front().GetIntensity().depth;
			std::cout
				<< std::format(std::locale(""), "{:8} | {:5} | {}/pos | {:11L} N/s | {:14L} | {:0.2f} s\n",
					e, d, short_time_format(duration / size), std::size_t(nodes / duration.count()), nodes, duration.count());
			index++;
		}
	});

	reporter.Start();
	ex->Process();
	reporter.Force();

	return 0;*/
}