#include "benchmark/benchmark.h"
#include "IO/IO.h"
#include "Search/Search.h"
#include <cstdint>
#include <vector>
#include <iostream>

class TimingTable : public Table
{
public:
	TimingTable()
		: Table("Algorithm |Empties| Depth  |   TTS    |   Nodes/s   |     Nodes     ", "{:^10}|{:>6} |{:>7} | {:>8} |{:>12L} |{:>15L}")
	{}

	void PrintRow(std::string algorithm, int empties, Intensity intensity, int sample_size, uint64_t nodes, std::chrono::duration<double> duration)
	{
		Table::PrintRow(
			algorithm,
			empties,
			to_string(intensity),
			sample_size ? ShortTimeString(duration / sample_size) : "",
			std::size_t(nodes / duration.count()),
			nodes,
			sample_size,
			duration
		);
	}
};

int main(int argc, char* argv[])
{
	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv))
		return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();

	std::locale::global(std::locale(""));

	std::vector<ScoredPosition> data = ScoredPositions(LoadScoredGameFile("..\\data\\Edax4.4_selfplay\\level_20_from_e54.gs"));
	PatternBasedEstimator estimator = LoadPatternBasedEstimator("G:\\Reversi2\\iteration6.model");
	RAM_HashTable tt{ 10'000'000 };
	NegaMax nega_max;
	AlphaBeta alpha_beta;
	PVS pvs{ tt, estimator };
	MTD mtd{ pvs };
	IDAB idab{ mtd };

	auto solver_nega_max = Solver{ nega_max, /*parallel*/ true, /*silent*/ true };
	auto solver_alpha_beta = Solver{ alpha_beta, /*parallel*/ true, /*silent*/ true };
	auto solver_pvs = Solver{ pvs, /*parallel*/ true, /*silent*/ true };
	auto solver_mtd = Solver{ mtd, /*parallel*/ true, /*silent*/ true };
	auto solver_idab = Solver{ idab, /*parallel*/ true, /*silent*/ true };

	TimingTable table;
	table.PrintHeader();

	//for (int depth = 0; depth <= 10; depth++)
	//{
	//	int empty_count = 40;
	//	auto filtered_data = EmptyCountFiltered(data, empty_count);

	//	for (ConfidenceLevel level : { 1.0f, inf })
	//	{
	//		solver_pvs.Clear();
	//		solver_pvs.Solve(filtered_data, { min_score, max_score }, depth, level);
	//		table.PrintRow("PVS", empty_count, depth, level, filtered_data.size(), solver_pvs.nodes, solver_pvs.time);

	//		solver_mtd.Clear();
	//		solver_mtd.Solve(filtered_data, { min_score, max_score }, depth, level);
	//		table.PrintRow("MTD", empty_count, depth, level, filtered_data.size(), solver_mtd.nodes, solver_mtd.time);

	//		solver_idab.Clear();
	//		solver_idab.Solve(filtered_data, { min_score, max_score }, depth, level);
	//		table.PrintRow("IDAB", empty_count, depth, level, filtered_data.size(), solver_idab.nodes, solver_idab.time);

	//		table.PrintSeparator();
	//	}
	//}
	//table.PrintSeparator();

	for (int8_t empty_count = 0; empty_count <= 40; empty_count++)
	{
		auto filtered_data = Sample(100, EmptyCountFiltered(data, empty_count), /*seed*/ 31415);

		std::vector<ConfidenceLevel> levels{ inf };
		//if (empty_count > 15)
		//	levels = { 1.0f, inf };
		for (ConfidenceLevel level : levels)
		{
			Intensity intensity{ empty_count, level };
			//if (empty_count <= 10) {
			//	solver_nega_max.Clear();
			//	solver_nega_max.Solve(filtered_data, { min_score, max_score }, empty_count, level);
			//	table.PrintRow("NegaMax", empty_count, empty_count, level, filtered_data.size(), solver_nega_max.nodes, solver_nega_max.time);
			//}
			//if (empty_count <= 14) {
			//	solver_alpha_beta.Clear();
			//	solver_alpha_beta.Solve(filtered_data, { min_score, max_score }, empty_count, level);
			//	table.PrintRow("AlphaBeta", empty_count, empty_count, level, filtered_data.size(), solver_alpha_beta.nodes, solver_alpha_beta.time);
			//}
			//solver_pvs.Clear();
			//solver_pvs.Solve(filtered_data, { min_score, max_score }, empty_count, level);
			//table.PrintRow("PVS", empty_count, empty_count, level, filtered_data.size(), solver_pvs.nodes, solver_pvs.time);

			//solver_mtd.Clear();
			//solver_mtd.Solve(filtered_data, { min_score, max_score }, empty_count, level);
			//table.PrintRow("MTD", empty_count, empty_count, level, filtered_data.size(), solver_mtd.nodes, solver_mtd.time);

			solver_idab.Clear();
			solver_idab.Solve(filtered_data, { min_score, max_score }, intensity);
			table.PrintRow("IDAB", empty_count, intensity, filtered_data.size(), solver_idab.nodes, solver_idab.time);

			//table.PrintSeparator();
		}
	}

	return 0;
}
