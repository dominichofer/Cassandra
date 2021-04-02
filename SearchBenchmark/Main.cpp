#include "Core/Core.h"
#include "Search/Search.h"
#include "IO/IO.h"

#include <vector>
#include <iostream>
#include <execution>

int main(int argc, char* argv[])
{
	int random_seed_1 = 4182;
	int random_seed_2 = 5899;
	int size = 100;

	ProjectDB db;
	std::mutex cout_mtx;
	for (int e = 15; e < 60; e++)
	{
		Project proj([e ,&cout_mtx](const std::vector<Puzzle>& puzzle){
			std::chrono::duration<double> duration{0};
			uint64 nodes = 0;
			for (const auto& p : puzzle) {
				duration += p.Duration();
				nodes += p.Nodes();
			}
			std::scoped_lock lock{ cout_mtx };
			std::cout
				<< std::setw(8) << e << " | "
				<< short_time_format(duration / puzzle.size()) << "/pos | "
				<< std::setw(11) << std::size_t(nodes / duration.count()) << " N/s | "
				<< duration.count() <<"\n";
		});

		PosGen::RandomPlayed rnd(random_seed_1 + e, random_seed_2 + e, e);
		std::generate_n(
			std::back_inserter(proj),
			size,
			[&rnd]() { return Puzzle::Exact(rnd()); }
		);

		db.push_back(std::move(proj));
	}

	std::cout
		<< " Empties |     TTS     |      Speed      | Duration \n"
		<< "---------+-------------+-----------------+----------\n";
	std::cout.imbue(std::locale(""));

	PatternEval pattern_eval = DefaultPatternEval();
	HashTablePVS tt{ 1'000'000 };
	db.Solve(std::execution::par, IDAB{ tt, pattern_eval });
	return 0;
}