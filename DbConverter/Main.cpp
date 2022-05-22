#include "IO/IO.h"
#include "Core/Core.h"
#include "Search/Search.h"
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>
//#include <windows.h>

//BOOL WINAPI consoleHandler(DWORD signal) {
//
//	if (signal == CTRL_C_EVENT)
//	{
//		#pragma omp critical
//		{
//			std::cout << "Saving... ";
//			Save(file, proj);
//			std::cout << "done!\n";
//			std::terminate();
//		}
//	}
//	return TRUE;
//}

void PrintHistogram(const range<Puzzle> auto& puzzles)
{
	std::map<std::pair<int, std::vector<Intensity>>, int> hist;
	for (const Puzzle& p : puzzles)
	{
		std::set<Intensity> set = p.SolvedIntensities();
		std::vector<Intensity> ints{ set.begin(), set.end() };
		std::sort(ints.begin(), ints.end());
		hist[std::make_pair(p.EmptyCount(), std::move(ints))]++;
	}
	for (const auto& h : hist)
		std::cout << "e" << h.first.first << " : " << to_string(h.first.second) << " : " << h.second << '\n';
}

int main()
{
	std::cout.imbue(std::locale(""));
	//HashTablePVS tt{ 100'000'000 };
	//AAGLEM evaluator = DefaultPatternEval();
	//DataBase<Puzzle> puzzles = LoadEvalFit();

	//Process(std::execution::par,
	//	puzzles | std::views::filter([](const Puzzle& p) { return p.EmptyCount() <= 20; }),
	//	[&](Puzzle& p, std::size_t index) {
	//		p.insert(Request::ExactScore(p.pos));
	//		p.Solve(IDAB{ tt, evaluator });
	//	});
	//puzzles.WriteBack();


	for (int i = 0; i <= 0; i++)
	{
		DataBase<Puzzle> puzzles{ fmt::format(R"(G:\Reversi\play{}{}_eval_fit.puz)", i, i) };
		std::map<std::pair<int, int>, int> hist;
		for (const Puzzle& p : puzzles)
		{
			hist[std::make_pair(p.EmptyCount(), p.MaxSolvedIntensityScore().value_or(-99))]++;
		}
		for (int e = 0; e <= 60; e++)
		for (int i = -32; i <= +32; i++)
			std::cout << e << " : " << i << " : " << hist[std::make_pair(e,i)] << '\n';
	}
	return 0;

	std::cout << "eval_fit\n";
	PrintHistogram(LoadEvalFit());

	std::cout << "accuracy_fit\n";
	PrintHistogram(LoadAccuracyFit());

	std::cout << "move_sort\n";
	PrintHistogram(LoadMoveSort());
	return 0;
}