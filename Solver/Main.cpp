#include "Core/Core.h"
#include "Pattern/Pattern.h"
#include "Search/Search.h"
#include "IO/IO.h"

void PrintHelp()
{
	std::cout
		<< "   -w <int> <int>     Window, bounds not included\n"
		<< "   -d <int[@float]>   Depth [Confidence Level]\n"
		<< "   -m <file>          Model\n"
		<< "   -tt <int>          Buckets in Transposition Table\n"
		<< "   -solve <file>      Solves all positions in file.\n"
		<< "   -test <file>       Tests all positions in file.\n"
		<< "   -h                 Prints this help."
		<< std::endl;
}

int main(int argc, char* argv[])
{
	std::locale::global(std::locale(""));
	
	OpenInterval window{ -inf_score, +inf_score };
	int depth = 64;
	float confidence_level = std::numeric_limits<float>::infinity();
	std::string model;
	std::size_t buckets = 10'000'000;
	std::string file;
	bool solve = false;
	bool test = false;

	for (int i = 0; i < argc; i++)
	{
		auto arg = std::string(argv[i]);
		if (arg == "-w") {
			int lower = std::stoi(argv[++i]);
			int upper = std::stoi(argv[++i]);
			window = OpenInterval(lower, upper);
		}
		else if (arg == "-d") {
			arg = std::string(argv[++i]);
			std::tie(depth, confidence_level) = DepthClFromString(arg);
		}
		else if (arg == "-m") model = std::string(argv[++i]);
		else if (arg == "-tt") buckets = std::stoull(argv[++i]);
		else if (arg == "-solve") { file = argv[++i]; solve = true; }
		else if (arg == "-test") { file = argv[++i]; test = true; }
		else if (arg == "-h") { PrintHelp(); return 0; }
	}

	PatternBasedEstimator evaluator = LoadPatternBasedEstimator(model);
	HT tt{ buckets };
	PVS pvs{ tt, evaluator };
	IDAB idab{ pvs };
	Solver solver{ idab };
	if (solve)
		solver.Solve(LoadPosFile(file));
	else if (test)
		solver.Solve(LoadPosScoreFile(file));
	return 0;
}
