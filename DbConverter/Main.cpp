#include "IO/IO.h"
#include "Core/Core.h"
#include "Search/Search.h"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>

int main()
{
	PatternEval pattern_eval = DefaultPatternEval();
	HashTablePVS tt{ 100'000'000 };
	for (int e = 13; e < 50; e++)
	{
		auto pos = Load<Position>(R"(G:\Reversi\rnd_1k\e)" + std::to_string(e) + ".psc");
		const auto start = std::chrono::high_resolution_clock::now();
		#pragma omp parallel for
		for (int i = 0; i < pos.size(); i++)
			IDAB{ tt, pattern_eval }.Score(pos[i]);
		const auto stop = std::chrono::high_resolution_clock::now();
		std::cout << e << ": " <<  std::chrono::duration_cast<std::chrono::milliseconds>(stop - start) << std::endl;
	}
	return 0;
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