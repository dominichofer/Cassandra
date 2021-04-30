#include "Core/Core.h"
#include "Search/Search.h"
#include "IO/IO.h"
#include "IO/Integers.h"
#include "Math/Algorithm.h"
#include "Math/Statistics.h"
#include "Pattern/Evaluator.h"

#include <chrono>
#include <numeric>
#include <vector>
#include <iostream>
#include <iomanip>
#include <omp.h>

void Test(std::vector<Puzzle>& puzzles)
{
	HashTablePVS tt{ 1'000'000 };
	auto start = std::chrono::high_resolution_clock::now();
	for (auto& puzzle : puzzles)
		puzzle.Result();
}

void print(std::size_t index, const Search::Result& eval, int correct, std::chrono::nanoseconds duration, uint64 node_count)
{
	std::locale locale("");
	std::cout.imbue(locale);
	std::cout << std::setw(3) << std::to_string(index) << "|";
	std::cout << std::setw(6) << to_string(eval.intensity) << "|";
	std::cout << " " << DoubleDigitSignedInt(eval.window.lower()) << " |";
	std::cout << " " << DoubleDigitSignedInt(correct) << " |";
	if (eval.window.lower() == correct)
		std::cout << "     |";
	else
		std::cout << " " << DoubleDigitSignedInt(eval.window.lower() - correct) << " |";
	std::cout << std::setw(16) << std::chrono::duration_cast<std::chrono::milliseconds>(duration) << "|";
	std::cout << std::setw(16) << node_count << "|";

	if (duration.count() > 0)
		std::cout << std::setw(12) << static_cast<std::size_t>(static_cast<long double>(node_count) * std::nano::den / duration.count());
	std::cout << std::endl;
}

//void print(std::size_t index, int depth, int eval, int correct, std::chrono::nanoseconds duration, uint64 node_count)
void PrintPuzzle(const Puzzle& puzzle)
{
	static int index = 0;
	#pragma omp critical
	{
		index++;

		std::locale locale("");
		std::cout.imbue(locale);
		std::cout << std::setw(3) << std::to_string(index) << "|";
		std::cout << std::setw(6) << to_string(puzzle.Result().intensity) << "|";
		std::cout << " " << DoubleDigitSignedInt(puzzle.Score()) << " |";
		std::cout << " " << DoubleDigitSignedInt(0) << " |";
		//if (eval == correct)
			std::cout << "     |";
		//else
		//	std::cout << " " << DoubleDigitSignedInt(eval - correct) << " |";
		std::cout << std::setw(16) << std::chrono::duration_cast<std::chrono::milliseconds>(puzzle.Duration()) << "|";
		std::cout << std::setw(16) << puzzle.Nodes() << "|";

		if (puzzle.Duration().count() > 0)
			std::cout << std::setw(12) << static_cast<std::size_t>(static_cast<long double>(puzzle.Nodes()) / puzzle.Duration().count());
		std::cout << std::endl;
	}
}

int main(int argc, char* argv[])
{
	PatternEval pattern_eval = DefaultPatternEval();
	HashTablePVS tt{ 10'000'000 };
	IDAB algorithm{ tt, pattern_eval };

	//uint64 node_count = 0;
	//std::chrono::nanoseconds duration{ 0 };
	//std::vector<int> score_diff;

	std::cout << " # | depth| eval|score| diff|       time (s) |      nodes (N) |    N/s     \n";
	std::cout << "---+------+-----+-----+-----+----------------+----------------+------------\n";

	Project project = FForum_2;
	//std::vector<Puzzle> puzzles;
	//for (int i = 0; i <= 50; i+=2)
	//	puzzles.emplace_back(Position::Start(), Search::Request(i, 1.0_sigmas, OpenInterval::Whole()));
	//puzzles.emplace_back(Position::Start(), Search::Request(60, 1.0_sigmas, OpenInterval::Whole()));
	//Project project(puzzles);
	project.SetPuzzleCompletionTask(PrintPuzzle);

	project.SolveAll(std::execution::seq, algorithm, true);

	std::cout << "---+------+-----+-----+-----+----------------+----------------+------------\n";

	std::cout << project.Nodes() << " nodes in " << std::chrono::duration_cast<std::chrono::milliseconds>(project.Duration());
	if (project.Duration().count())
		std::cout << " (" << static_cast<std::size_t>(project.Nodes() / project.Duration().count()) << " N/s)";
	std::cout << '\n';

	//const auto correct = std::count_if(score_diff.begin(), score_diff.end(), [](int i) { return i == 0; });
	//std::cout << "Tests correct: " << correct << "\n";
	//std::cout << "Tests wrong: " << score_diff.size() - correct << "\n";
	//std::cout << "stddev(score_diff) = " << StandardDeviation(score_diff) << std::endl;

	std::cout << "TT LookUps: " << tt.LookUpCounter() << " Hits: " << tt.HitCounter() << " Updates: " << tt.UpdateCounter() << std::endl;

	return 0;
}

