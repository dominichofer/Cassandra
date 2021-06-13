#include "Core/Core.h"
#include "Search/Search.h"
#include "IO/IO.h"
#include "IO/Integers.h"
#include "Math/Algorithm.h"
#include "Math/Statistics.h"
#include "Pattern/Evaluator.h"
#include "PatternFit/PatternFit.h"

#include <chrono>
#include <numeric>
#include <vector>
#include <iostream>
#include <iomanip>
#include <omp.h>

//void Test(std::vector<Puzzle>& puzzles)
//{
//	HashTablePVS tt{ 1'000'000 };
//	auto start = std::chrono::high_resolution_clock::now();
//	for (auto& puzzle : puzzles)
//		puzzle.Result();
//}

//void print(std::size_t index, int depth, int eval, int correct, std::chrono::nanoseconds duration, uint64 node_count)

void PrintFooter()
{
	std::cout << "---+------+-----+----------------+----------------+------------\n";
}

void PrintFooterTest()
{
	std::cout << "---+------+-----+-----+-----+----------------+----------------+------------\n";
}

void PrintHeader()
{
	std::cout << " # | depth| eval|       time (s) |        nodes (N) |    N/s     \n";
	std::cout << "---+------+-----+----------------+------------------+------------\n";
}

void PrintHeaderTest()
{
	std::cout << " # | depth| eval|score| diff|       time (s) |        nodes (N) |    N/s     \n";
	std::cout << "---+------+-----+-----+-----+----------------+------------------+------------\n";
}

void PrintPuzzle(const Puzzle& puzzle, int index)
{
	#pragma omp critical
	{
		auto task = puzzle.MaxIntensity();
		int score = task.Score();

		std::cout << std::setw(3) << std::to_string(index) << "|";
		std::cout << std::setw(6) << to_string(task.Intensity()) << "|";
		std::cout << " " << DoubleDigitSignedInt(score) << " |";
		std::cout << std::setw(16) << std::chrono::duration_cast<std::chrono::milliseconds>(puzzle.Duration()) << "|";
		std::cout << std::setw(18) << puzzle.Nodes() << "|";

		if (puzzle.Duration().count())
			std::cout << std::setw(12) << static_cast<std::size_t>(puzzle.Nodes() / puzzle.Duration().count());
		std::cout << std::endl;
	}
}

void PrintPuzzleTest(const Puzzle& puzzle, int index)
{
#pragma omp critical
	{
		int score = puzzle.tasks[0].Score();
		int eval = puzzle.tasks[1].Score();

		std::cout << std::setw(3) << std::to_string(index) << "|";
		std::cout << std::setw(6) << to_string(puzzle.tasks[1].Intensity()) << "|";
		std::cout << " " << DoubleDigitSignedInt(eval) << " |";
		std::cout << " " << DoubleDigitSignedInt(score) << " |";
		if (eval == score)
			std::cout << "     |";
		else
			std::cout << " " << DoubleDigitSignedInt(eval - score) << " |";
		std::cout << std::setw(16) << std::chrono::duration_cast<std::chrono::milliseconds>(puzzle.Duration()) << "|";
		std::cout << std::setw(16) << puzzle.Nodes() << "|";

		if (puzzle.Duration().count())
			std::cout << std::setw(12) << static_cast<std::size_t>(puzzle.Nodes() / puzzle.Duration().count());
		std::cout << std::endl;
	}
}

int main(int argc, char* argv[])
{
	FitAccuracyModel();
	return 0;
	//for (int e = 0; e <= 50; e++)
	//{
	//	auto puzzles = Load<std::vector<Puzzle>>(R"(G:\Reversi\rnd\e)" + std::to_string(e) + ".puz");
	//	if (e > 24)
	//		for (auto& p : puzzles)
	//			p.clear();
	//	Save(R"(G:\Reversi\rnd_100k\e)" + std::to_string(e) + ".puz", puzzles.begin(), puzzles.begin() + 100'000);
	//}
	//return 0;
	//for (int e = 25; e <= 50; e++)
	//{
	//	auto puzzles = Load<std::vector<Puzzle>>(R"(G:\Reversi\rnd_100k\e)" + std::to_string(e) + ".puz");
	//	Save(R"(G:\Reversi\rnd_100k\e)" + std::to_string(e) + ".puz", puzzles);
	//}
	PatternEval pattern_eval = DefaultPatternEval();
	HashTablePVS tt{ 1'000'000'000 };

	//uint64 node_count = 0;
	//std::chrono::nanoseconds duration{ 0 };

	std::locale locale("");
	std::cout.imbue(locale);

	//std::string file_name = R"(G:\Reversi\rnd\e26.psc)";
	//auto pos_score = Load<PosScore>(file_name);
	//std::vector<Puzzle> puzzles;
	//for (const auto& ps : pos_score)
	//{
	//	auto p = Puzzle::Exact(ps.pos, ps.score);
	//	p.Add(ps.pos.EmptyCount(), 1.1_sigmas);
	//	puzzles.push_back(std::move(p));
	//}
	//Project project(puzzles);

	//{
	//	std::vector<Puzzle> proj = FForum_3;
	//	PrintHeaderTest();
	//	Process(std::execution::seq, proj,
	//		[&](Puzzle& p, std::size_t index) {
	//			p.insert(Request::ExactScore(p.pos));
	//			p.Solve(IDAB{ tt, pattern_eval });
	//			PrintPuzzleTest(p, index);
	//		});

	//	auto nodes = Nodes(proj);
	//	auto duration = Duration(proj);

	//	PrintFooter();
	//	std::cout << nodes << " nodes in " << std::chrono::duration_cast<std::chrono::milliseconds>(duration);
	//	if (duration.count())
	//		std::cout << " (" << static_cast<std::size_t>(nodes / duration.count()) << " N/s)";
	//	std::cout << "\n\n";
	//}
	//return 0;


	//for (int e = 51; e <= 60; e++)
	//{
	//	auto file = R"(G:\Reversi\rnd_100k\e)" + std::to_string(e) + ".puz";
	//	std::vector<Puzzle> proj = Load<std::vector<Puzzle>>(file);

	//	PrintHeader();
	//	Process(std::execution::par, proj,
	//		[&](Puzzle& p, std::size_t index) {
	//			p.insert(Request(10, 1.1_sigmas));
	//			bool had_work = p.Solve(IDAB{ tt, pattern_eval });
	//			if (had_work)
	//				PrintPuzzle(p, index);
	//		});

	//	auto nodes = Nodes(proj);
	//	auto duration = Duration(proj);

	//	PrintFooter();
	//	std::cout << nodes << " nodes in " << std::chrono::duration_cast<std::chrono::milliseconds>(duration);
	//	if (duration.count())
	//		std::cout << " (" << static_cast<std::size_t>(nodes / duration.count()) << " N/s)";
	//	std::cout << "\n\n";

	//	Save(file, proj);
	//	std::cout << "Run 1, EmptyCount: " << e << "\n";
	//}

	//std::vector<Puzzle> puzzles;
	//for (int e = 0; e <= 50; e++)
	//{
	//	for (const auto& pos : PosGen::RandomPlayed{ e }(200))
	//		puzzles.push_back(pos);
	//}
	//Save(R"(G:\Reversi\model_eval.puz)", puzzles);
	//return 0;
	auto puzzles = Load<std::vector<Puzzle>>(R"(G:\Reversi\model_eval.puz)");
	PrintHeader();
	Process(std::execution::par, puzzles,
		[&](Puzzle& p, std::size_t index) {
			p.insert(Request::ExactScore(p.pos));
			bool had_work = p.Solve(IDAB{ tt, pattern_eval });
			PrintPuzzle(p, index);
			if (had_work) {
				#pragma omp critical
				{
					Save(R"(G:\Reversi\model_eval.puz)", puzzles);
				}
			}
		});
	Save(R"(G:\Reversi\model_eval.puz)", puzzles);
	return 0;

	for (int e = 20; e <= 50; e++)
	{
		auto file = R"(G:\Reversi\rnd_100k\e)" + std::to_string(e) + ".puz";
		std::vector<Puzzle> proj = Load<std::vector<Puzzle>>(file);

		PrintHeader();
		Process(std::execution::par, proj,
			[&](Puzzle& p, std::size_t index) {
				p.insert(Request(p.pos.EmptyCount() - 10, 1.1_sigmas));
				bool had_work = p.Solve(IDAB{ tt, pattern_eval });
				if (had_work)
					PrintPuzzle(p, index);
			});

		auto nodes = Nodes(proj);
		auto duration = Duration(proj);

		PrintFooter();
		std::cout << nodes << " nodes in " << std::chrono::duration_cast<std::chrono::milliseconds>(duration);
		if (duration.count())
			std::cout << " (" << static_cast<std::size_t>(nodes / duration.count()) << " N/s)";
		std::cout << "\n\n";

		Save(file, proj);
		std::cout << "Run 1, EmptyCount: " << e << "\n";
	}
	

	//while (not project.IsSolved())
	//{
	//	#pragma omp parallel for
	//	for (int i = 0; i < 1000; i++)
	//		project.SolveNext(algorithm);

	//	for (int i = 0; i < pos_score.size(); i++)
	//		if (project[i].IsSolved())
	//			pos_score[i].score = project[i].Score();
	//	Save(file_name, pos_score);
	//	std::cout << "Saved!" << std::endl;
	//}

	// Postprocessing
	//std::vector<int> score_diff;
	//for (const Puzzle& puzzle : project.Puzzles())
	//	score_diff.push_back(puzzle.Result(0).Score() - puzzle.Result().Score());
	//const auto correct = std::count_if(score_diff.begin(), score_diff.end(), [](int i) { return i == 0; });

	//std::cout << "---+------+-----+-----+-----+----------------+----------------+------------\n";

	//std::cout << project.Nodes() << " nodes in " << std::chrono::duration_cast<std::chrono::milliseconds>(project.Duration());
	//if (project.Duration().count())
	//	std::cout << " (" << static_cast<std::size_t>(project.Nodes() / project.Duration().count()) << " N/s)";
	//std::cout << '\n';

	//std::cout << "Tests correct: " << correct << "\n";
	//std::cout << "Tests wrong: " << score_diff.size() - correct << "\n";
	//std::cout << "stddev(score_diff) = " << StandardDeviation(score_diff) << std::endl;

	std::cout << "TT LookUps: " << tt.LookUpCounter() << " Hits: " << tt.HitCounter() << " Updates: " << tt.UpdateCounter() << std::endl;

	return 0;
}

