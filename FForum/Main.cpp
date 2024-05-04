#include "Pattern/Pattern.h"
#include "Search/Search.h"
#include "IO/IO.h"
#include <iostream>

int main(int argc, char* argv[])
{
	std::locale::global(std::locale(""));
	PatternBasedEstimator estimator = LoadPatternBasedEstimator("G:\\Cassandra\\iteration12.model");
	RAM_HashTable tt{ 10'000'000 };
	PVS pvs{ tt, estimator };
	MoveSorter move_sorter{ tt, pvs };
	//DTS dts{ tt, pvs, move_sorter, 5 };
	AspirationSearch asp{ pvs };
	IDAB idab{ asp };
	auto solver = Solver{ idab, false, 1 };

	solver.PrintHeader();

	//solver.Solve(LoadScoredPositionFile("..\\data\\Edax4.4_selfplay\\e20_level_20_from_e54.ps"));
	//solver.Solve(LoadScoredPositionFile("..\\data\\Edax4.4_selfplay\\e21_level_20_from_e54.ps"));
	//solver.Solve(LoadScoredPositionFile("..\\data\\Edax4.4_selfplay\\e22_level_20_from_e54.ps"));
	//solver.Solve(LoadScoredPositionFile("..\\data\\Edax4.4_selfplay\\e23_level_20_from_e54.ps"));
	//solver.Solve(LoadScoredPositionFile("..\\data\\Edax4.4_selfplay\\e24_level_20_from_e54.ps"));
	//solver.Solve(LoadScoredPositionFile("..\\data\\Edax4.4_selfplay\\e25_level_20_from_e54.ps"));

	solver.Solve(LoadScoredPositionFile("..\\data\\fforum-1-19.ps"));
	solver.Solve(LoadScoredPositionFile("..\\data\\fforum-20-39.ps"));
	solver.Solve(LoadScoredPositionFile("..\\data\\fforum-40-59.ps"));
	solver.Solve(LoadScoredPositionFile("..\\data\\fforum-60-79.ps"));

	return 0;
}
