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

void print(std::size_t index, unsigned int depth, int eval, int correct, std::chrono::nanoseconds duration, uint64 node_count)
{
	std::locale locale("");
	std::cout.imbue(locale);
	std::cout << std::setw(3) << std::to_string(index) << "|";
	std::cout << std::setw(6) << std::to_string(depth) << "|";
	std::cout << " " << DoubleDigitSignedInt(eval) << " |";
	std::cout << " " << DoubleDigitSignedInt(correct) << " |";
	if (eval == correct)
		std::cout << "     |";
	else
		std::cout << " " << DoubleDigitSignedInt(eval - correct) << " |";
	std::cout << std::setw(16) << std::chrono::duration_cast<std::chrono::milliseconds>(duration) << "|";
	std::cout << std::setw(16) << node_count << "|";

	if (duration.count() > 0)
		std::cout << std::setw(12) << static_cast<std::size_t>(static_cast<long double>(node_count) * std::nano::den / duration.count());
	std::cout << std::endl;
}

int main(int argc, char* argv[])
{
	PatternEval pattern_eval = DefaultPatternEval();
	HashTablePVS tt{ 10'000'000 };
	uint64 node_count = 0;
	std::chrono::nanoseconds duration{ 0 };
	std::vector<int> score_diff;

	std::cout << " # | depth| eval|score| diff|       time (s) |      nodes (N) |    N/s     \n";
	std::cout << "---+------+-----+-----+-----+----------------+----------------+------------\n";

	//Search::AlphaBetaFailSoft algorithm;
	auto start = std::chrono::high_resolution_clock::now();
	//#pragma omp parallel for reduction(+:node_count)
	for (int i = 20; i < 40; i++)
	{
		IDAB algorithm{ tt, pattern_eval };
		//tt.Clear();
		auto start = std::chrono::high_resolution_clock::now();
		Search::Result result = algorithm.Eval(FForum[i].pos, Search::Request::Exact(FForum[i].pos));
		//std::cout << to_string(algorithm.log) << std::endl;
		auto stop = std::chrono::high_resolution_clock::now();

		node_count += algorithm.node_count;
		duration += stop - start;
		score_diff.push_back(result.window.lower() - FForum[i].score);

		print(i, result, FForum[i].score, stop - start, algorithm.node_count);
	}
	auto stop = std::chrono::high_resolution_clock::now();

	std::cout << "---+------+-----+-----+-----+----------------+----------------+------------\n";

	const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
	std::cout << node_count << " nodes in " << ms;
	if (ms.count() > 0)
		std::cout << " (" << static_cast<std::size_t>(node_count * std::milli::den / ms.count()) << " N/s)";
	std::cout << '\n';

	const auto correct = std::count_if(score_diff.begin(), score_diff.end(), [](int i) { return i == 0; });;
	std::cout << "Tests correct: " << correct << "\n";
	std::cout << "Tests wrong: " << score_diff.size() - correct << "\n";
	std::cout << "stddev(score_diff) = " << StandardDeviation(score_diff) << std::endl;

	std::cout << "TT LookUps: " << tt.LookUpCounter() << " Hits: " << tt.HitCounter() << " Updates: " << tt.UpdateCounter() << std::endl;

	return 0;
}

