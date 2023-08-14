#include "Pattern/Pattern.h"
#include "Search/Search.h"
#include "IO/IO.h"
#include <iostream>

int main(int argc, char* argv[])
{
	std::locale::global(std::locale(""));
	PatternBasedEstimator estimator = LoadPatternBasedEstimator("G:\\Reversi2\\iteration3.model");
	HT tt{ 10'000'000 };
	PVS pvs{ tt, estimator };
	IDAB idab{ pvs };
	auto solver = Solver{ idab, false };

	solver.PrintHeader();
	solver.Solve(LoadPosScoreFile("..\\data\\fforum-1-19.ps"));
	solver.PrintSummary(); std::cout << std::endl;

	solver.PrintHeader();
	solver.Solve(LoadPosScoreFile("..\\data\\fforum-20-39.ps"));
	solver.PrintSummary(); std::cout << std::endl;

	solver.PrintHeader();
	solver.Solve(LoadPosScoreFile("..\\data\\fforum-40-59.ps"));
	solver.PrintSummary(); std::cout << std::endl;

	solver.PrintHeader();
	solver.Solve(LoadPosScoreFile("..\\data\\fforum-60-79.ps"));
	solver.PrintSummary();

	return 0;
}
