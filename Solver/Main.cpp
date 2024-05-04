#include "Base/Base.h"
#include "IO/IO.h"
#include "Pattern/Pattern.h"
#include "Search/Search.h"

void PrintHelp()
{
	std::cout
		<< "   -m <file>          Model (required)\n"
		<< "   -w <int> <int>     Window, bounds not included\n"
		<< "   -d <int[@float]>   Depth [Confidence Level]\n"
		<< "   -tt <int>          Transposition Table size\n"
		<< "   -t <int>           Threads\n"
		<< "   -solve <file>      Solves all positions in file.\n"
		<< "   -test <file>       Tests all positions in file.\n"
		<< "   -h                 Prints this help."
		<< std::endl;
}

int main(int argc, char* argv[])
{
	std::locale::global(std::locale(""));
	
	std::string model;
	OpenInterval window{ min_score, max_score };
	Intensity intensity{ 64 };
	std::size_t tt_size = 10'000'000;
	int threads = 1;
	std::string file;
	bool solve = false;
	bool test = false;

	for (int i = 0; i < argc; i++)
	{
		auto arg = std::string(argv[i]);
		if (arg == "-m") model = std::string(argv[++i]);
		else if (arg == "-w") {
			int lower = std::stoi(argv[++i]);
			int upper = std::stoi(argv[++i]);
			window = OpenInterval{ lower, upper };
		}
		else if (arg == "-d") intensity = Intensity::FromString(std::string_view(argv[++i]));
		else if (arg == "-tt") tt_size = std::stoull(argv[++i]);
		else if (arg == "-t") threads = std::stoi(argv[++i]);
		else if (arg == "-solve") { file = argv[++i]; solve = true; }
		else if (arg == "-test") { file = argv[++i]; test = true; }
		else if (arg == "-h") { PrintHelp(); return 0; }
	}

	PatternBasedEstimator evaluator = LoadPatternBasedEstimator(model);
	RAM_HashTable tt{ tt_size };
	PVS pvs{ tt, evaluator };
	MTD mtd{ pvs };
	IDAB idab{ mtd };
	Solver solver{ idab, false, threads };
	if (solve)
		solver.Solve(LoadPositionFile(file), window, intensity);
	else if (test)
		solver.Solve(LoadScoredPositionFile(file), window, intensity);
	return 0;
}
