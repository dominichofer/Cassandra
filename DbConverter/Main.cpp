#include "IO/IO.h"
#include "Core/Core.h"
#include "Search/Search.h"
#include <iostream>
#include <string>
#include <vector>
//#include <windows.h>

void PrintPuzzle(const Puzzle& puzzle, int index)
{
	if (index % 1000 == 0)
	{
		#pragma omp critical
		{
			for (const Puzzle::Task& task : puzzle.tasks)
			{
				std::cout
					<< std::setw(3) << std::to_string(index) << "|"
					<< std::setw(6) << to_string(task.Intensity()) << "|"
					<< " " << DoubleDigitSignedInt(task.Score()) << " |"
					<< std::setw(16) << std::chrono::duration_cast<std::chrono::milliseconds>(puzzle.Duration()) << "|"
					<< std::setw(16) << puzzle.Nodes() << "|";
				if (puzzle.Duration().count())
					std::cout << std::setw(12) << static_cast<std::size_t>(puzzle.Nodes() / puzzle.Duration().count());
				std::cout << '\n';
			}
		}
	}
}

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

template <typename T>
auto Split(std::vector<T> vec, std::size_t size)
{
	std::vector<T> rest;
	rest.reserve(vec.size() - size);
	std::move(vec.begin() + size, vec.end(), std::back_inserter(rest));
	return std::make_tuple(vec, rest);
}

int main()
{
	std::locale locale("");
	std::cout.imbue(locale);

	PatternEval pattern_eval = DefaultPatternEval();
	HashTablePVS tt{ 1'000'000'000 };

	DataBase<Puzzle> DB;
	for (int e = 0; e <= 60; e++)
		DB.Add(R"(G:\Reversi\rnd_100k\e)" + std::to_string(e) + ".puz");

	for (Puzzle& puzzle : DB)
	{
		puzzle.RemoveAllUndone();
		puzzle.insert(Request{ puzzle.pos.EmptyCount() - 10, 1.1_sigmas });
	}

	Process(std::execution::par, DB,
		[&](Puzzle& p, std::size_t index) {
			bool had_work = p.Solve(IDAB{ tt, pattern_eval });
			if (had_work)
				PrintPuzzle(p, index);
		});

	DB.WriteBack();

	std::vector<Position> pos;
	std::vector<float> score;
	for (Puzzle& puzzle : DB)
	{
		auto result = puzzle.ResultOf(Request{ puzzle.pos.EmptyCount() - 10, 1.1_sigmas });
		pos.push_back(puzzle.pos);
		score.push_back(result.score);
	}
	auto [test_pos, train_pos] = Split(pos, 50'000);
	auto [test_score, train_score] = Split(score, 50'000);

	return 0;
	//for (int i = 0; i <= 60; i++)
	//{
	//	std::vector<PosScore> score = LoadVec_old<PosScore>(R"(G:\Reversi\rnd\e)" + std::to_string(i) + ".psc");
	//	Save(R"(G:\Reversi\rnd\e)" + std::to_string(i) + ".w", score);
	//}

	//return 0;
	//for (int e = 26; e <= 26; e++)
	//{
	//	PuzzleProject proj;
	//	if (e == 25)
	//	{
	//		for (Position pos : PosGen::RandomPlayed{ 25 }(1'000'000))
	//			proj.push_back(pos);
	//	}
	//	else
	//	{
	//		auto PS = LoadVec_old<PosScore>(R"(G:\Reversi\rnd\e)" + std::to_string(e) + ".psc");
	//		for (const PosScore& ps : PS)
	//			if (ps.score == -99)
	//				proj.push_back(ps.pos);
	//			else
	//				proj.push_back(Puzzle::WithExactScore(ps.pos, ps.score / 2));
	//	}
	//	Save(R"(G:\Reversi\rnd\e)" + std::to_string(e) + ".puz", proj);
	//}
	//return 0;

	//using namespace std::chrono_literals;

	//PatternEval pattern_eval = DefaultPatternEval();
	//HashTablePVS tt{ 100'000'000 };

	//for (int e = 26; e <= 30; e++)
	//{
	//	std::filesystem::path file = R"(G:\Reversi\rnd\e)" + std::to_string(e) + ".puz";
	//	PuzzleProject proj = Load<PuzzleProject>(file);
	//	proj.MakeAllHave(Request(e-10, 1.1_sigmas));
	//	Save(file, proj);

	//	//if (!SetConsoleCtrlHandler(consoleHandler, TRUE)) {
	//	//	std::cerr << "ERROR: Could not set console handler";
	//	//	return 1;
	//	//}

	//	std::jthread saver([&]() {
	//		while (true)
	//		{
	//			std::this_thread::sleep_for(60s);
	//			#pragma omp critical
	//			{
	//				std::cout << "Saving... ";
	//				Save(file, proj);
	//				std::cout << "done!\n";
	//			}
	//		}
	//		});

	//	Process(std::execution::par, proj,
	//		[&](Puzzle& p, std::size_t index) {
	//			p.Solve(IDAB{ tt, pattern_eval });
	//			PrintPuzzle(p, index);
	//		});
	//	Save(file, proj);

	//	auto nodes = Nodes(proj);
	//	auto duration = Duration(proj);

	//	std::cout << "---+------+-----+-----+-----+----------------+----------------+------------\n";
	//	std::cout << nodes << " nodes in " << std::chrono::duration_cast<std::chrono::milliseconds>(duration);
	//	if (duration.count())
	//		std::cout << " (" << static_cast<std::size_t>(nodes / duration.count()) << " N/s)";
	//}
	//return 0;

	/*std::vector<PuzzleProject> projects;
	for (int e = 25; e < 50; e++)
	{
		PuzzleProject proj;
		auto positions = Load<Position>(R"(G:\Reversi\rnd_1k\e)" + std::to_string(e) + ".pos");
		for (const Position& pos : positions)
			proj.push_back(Puzzle::Exact(pos));
		projects.push_back(std::move(proj));
	}

	Process(std::execution::par, projects,
		[&](Puzzle& p) { p.Solve(IDAB{ tt, pattern_eval }); },
		[&](const PuzzleProject& proj) {
			#pragma omp critical
			std::cout << proj.front().Position().EmptyCount() << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(Duration(proj)) << std::endl;
		});

	return 0;*/
	//for (int e = 0; e <= 50; e++)
	//{
	//	auto pos = PosGen::RandomPlayed{e}(1'000);
	//	Save(R"(G:\Reversi\rnd_1k\e)" + std::to_string(e) + ".psc", pos);

	//	std::fstream stream(R"(G:\Reversi\rnd_1k\e)" + std::to_string(e) + ".script", std::ios::out);
	//	if (!stream.is_open())
	//		throw;

	//	for (const auto& p : pos)
	//		stream << p << "\n";
	//}
	//return 0;
}