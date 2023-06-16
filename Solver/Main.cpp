#include "Core/Core.h"
#include "Pattern/Pattern.h"
#include "Search/Search.h"
#include "IO/IO.h"

void PrintHelp()
{
	std::cout
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
		if (arg == "-d") {
			arg = std::string(argv[++i]);
			depth = DepthFromString(arg);
			confidence_level = ConfidenceLevelFromString(arg);
		}
		else if (arg == "-m") model = std::string(argv[++i]);
		else if (arg == "-tt") buckets = std::stoull(argv[++i]);
		else if (arg == "-solve") { file = argv[++i]; solve = true; }
		else if (arg == "-test") { file = argv[++i]; test = true; }
		else if (arg == "-h") { PrintHelp(); return 0; }
	}

	std::vector<PosScore> data = LoadPosScoreFile(file);
	PatternBasedEstimator evaluator = LoadPatternBasedEstimator(model);
	HT tt{ buckets };
	PVS pvs(tt, evaluator);

	ResultTable table;
	table.PrintHeader();
	for (const PosScore& ps : data)
		table.PrintRow(pvs.Eval(ps.pos));
	table.PrintSeparator();
	table.PrintSummary();
	return 0;
}
