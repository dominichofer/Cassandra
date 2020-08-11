#include "Core/Core.h"
#include "Search/Search.h"
#include "IO/IO.h"
#include "IO/Integers.h"
#include "Math/Algorithm.h"

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
	{
		puzzle.Result();
	}
}

void print(std::size_t index, unsigned int depth, Score eval, Score correct, std::chrono::nanoseconds duration, std::size_t node_count)
{
	std::locale locale("");
	std::wcout.imbue(locale);
	std::wcout << std::setw(3) << std::to_wstring(index) << L"|";
	std::wcout << std::setw(6) << std::to_wstring(depth) << L"|";
	std::wcout << " " << DoubleDigitSignedInt(eval) << L" |";
	std::wcout << " " << DoubleDigitSignedInt(correct) << L" |";
	if (eval == correct)
		std::wcout << L"     |";
	else
		std::wcout << " " << DoubleDigitSignedInt(eval - correct) << L" |";
	std::wcout << std::setw(16) << time_format(duration) << L"|";
	std::wcout << std::setw(16) << node_count << L"|";

	if (duration.count() > 0)
		std::wcout << std::setw(12) << static_cast<std::size_t>(static_cast<long double>(node_count) * std::nano::den / duration.count());
	std::wcout << std::endl;
}

int main(int argc, char* argv[])
{
	HashTablePVS tt{ 1'000'000 };
	std::size_t node_count = 0;
	std::chrono::nanoseconds duration{ 0 };
	std::vector<int> score_diff;

	std::wcout << L" # | depth| eval|score| diff|       time (s) |      nodes (N) |    N/s     \n";
	std::wcout << L"---+------+-----+-----+-----+----------------+----------------+------------\n";

	Search::PV algorithm{ tt };
	//Search::AlphaBetaFailSoft algorithm;
	auto start = std::chrono::high_resolution_clock::now();
	////#pragma omp parallel for reduction(+:node_count)
	for (std::size_t i = 20; i < 40; i++)
	{
		auto start = std::chrono::high_resolution_clock::now();
		Puzzle puzzle(FForum[i].pos, Search::Intensity::Exact(FForum[i].pos));
		puzzle.Solve(algorithm);
		auto stop = std::chrono::high_resolution_clock::now();

		node_count += puzzle.Result().value().node_count;
		duration += stop - start;
		score_diff.push_back(puzzle.Result().value().window.lower() - FForum[i].score);

		print(i, puzzle.Intensity().depth, puzzle.Result().value().window.lower(), FForum[i].score, stop - start, puzzle.Result().value().node_count);
	}
	auto stop = std::chrono::high_resolution_clock::now();

	std::wcout << L"---+------+-----+-----+-----+----------------+----------------+------------\n";

	std::wcout << node_count << L" nodes in " << time_format(duration)
		<< L" (" << static_cast<std::size_t>(static_cast<long double>(node_count)* std::nano::den / duration.count()) << L" N/s)\n";

	const auto correct = std::count_if(score_diff.begin(), score_diff.end(), [](int i) { return i == 0; });;
	std::wcout << L"Tests correct: " << correct << "\n";
	std::wcout << L"Tests wrong: " << score_diff.size() - correct << "\n";
	std::wcout << L"stddev(score_diff) = " << SampleStandardDeviation(score_diff) << std::endl;

	return 0;
}

