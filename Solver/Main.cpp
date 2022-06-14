#include "Core/Core.h"
#include "CoreIO/CoreIO.h"
#include "Pattern/Pattern.h"
#include "PatternIO/PatternIO.h"
#include "Search/Search.h"
#include <iostream>

//class SolverTable : public Table
//{
//	bool test;
//public:
//	SolverTable(bool test) : test(test)
//	{
//		AddColumn("#", 2, "{:>2L}");
//		AddColumn("depth", 6, "{:6}");
//		AddColumn("eval", 5, " {:+03} ");
//		AddColumn("score", 5, " {:+03} ", test);
//		AddColumn("diff", 5, " {:+03} ", test);
//		AddColumn("time [s]", 16, "{:>#16.3f}");
//		AddColumn("nodes", 18, "{:>18L}");
//		AddColumn("nodes/s", 14, "{:>14.0Lf}");
//	}
//};


void PrintHelp()
{
	std::cout
		<< "   -d <int>        Depth\n"
		<< "   -c <float>      Confidence\n"
		<< "   -m <file>       Model\n"
		<< "   --tt <int>       Buckets in Transposition Table\n"
		<< "   --solve <file>  Solves all positions in file."
		<< "   -h              Prints this help."
		<< std::endl;

}

int main(int argc, char* argv[])
{
	std::locale::global(std::locale(""));
	
	auto intensity = Intensity::Exact();
	AAGLEM model;
	HT tt{ 10'000'000 };

	for (int i = 0; i < argc; i++)
	{
		if (std::string(argv[i]) == "-d") intensity.depth = std::stoi(argv[++i]);
		else if (std::string(argv[i]) == "-c") intensity.certainty = Confidence(std::stof(argv[++i]));
		else if (std::string(argv[i]) == "-m") model = Deserialize<AAGLEM>(argv[++i]);
		else if (std::string(argv[i]) == "--tt") tt = HT{ std::stoull(argv[++i]) };
		else if (std::string(argv[i]) == "-h") { PrintHelp(); return 0; }
	}
	
	Position pos;
	try
	{
		pos = ParsePosition_SingleLine(std::string(argv[argc - 2]) + ' ' + argv[argc - 1]);
	}
	catch (const std::exception& ex)
	{
		std::cerr << ex.what() << '\n';
	}

	auto alg = IDAB<PVS>{ tt, model };
	int score = alg.Eval(pos, intensity);
	std::cout << 2 * score << '\n';
	return 0;
}
