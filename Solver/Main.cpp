#include "Core/Core.h"
#include "CoreIO/CoreIO.h"
#include "Pattern/Pattern.h"
#include "PatternIO/PatternIO.h"
#include "Search/Search.h"
#include "SearchIO/SearchIO.h"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <iostream>
#include <optional>
#include <ranges>
#include <string>
#include <vector>

class SolverTable : public Table
{
	int row = 1;
public:
	SolverTable() : Table(
		" # | depth|score|     time [s] |    nodes (N)   |     N/s     | principal variation ",
		"{:>3}|{:6}  {:+03}  {:>#14.3f} {:>16L} {:>13.0Lf} {:<}")
	{}

	void PrintRow(Intensity intensity, int score, float time, uint64 nodes, Field best_move)
	{
		float nps = nodes / time;
		std::optional<float> opt_nps = time ? std::optional(nps) : std::nullopt;
		Table::PrintRow(row++, intensity, score * 2, time, nodes, opt_nps, best_move);
	}

	void PrintSummary(float time, uint64 nodes) const
	{
		float nps = nodes / time;
		fmt::print("                 {:>#14.3f} {:>16L} {:>13.0Lf}\n", time, nodes, nps);
	}
};

class TestTable : public Table
{
	int row = 1;
public:
	TestTable() : Table(
		" # | depth| eval|score|diff |     time [s] |    nodes (N)   |     N/s     | principal variation ",
		"{:>3}|{:6}  {:+03}   {:+03}   {:+03}  {:>#14.3f} {:>16L} {:>13.0Lf} {:<}")
	{}

	void PrintRow(Intensity intensity, int eval, int score, float time, uint64 nodes, Field best_move)
	{
		float nps = nodes / time;
		std::optional<int> opt_diff = (eval != score) ? std::optional((eval - score) * 2) : std::nullopt;
		std::optional<float> opt_nps = time ? std::optional(nps) : std::nullopt;
		Table::PrintRow(row++, intensity, eval * 2, score * 2, opt_diff, time, nodes, opt_nps, best_move);
	}

	void PrintSummary(double diff, float time, uint64 nodes) const
	{
		float nps = nodes / time;
		fmt::print("                       {:#5.2f} {:>#14.3f} {:>16L} {:>13.0Lf}\n", diff * 2, time, nodes, nps);
	}
};


class SolverLab
{
	SolverTable table;
public:
	void Run(std::ranges::range auto&& poss, Intensity intensity, Algorithm& alg)
	{
		table.PrintHeader();
		uint64 total_nodes = 0;
		double total_duration = 0;
		for (Position pos : poss)
		{
			auto start = std::chrono::high_resolution_clock::now();
			auto result = alg.Eval(pos, intensity);
			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = round<std::chrono::milliseconds>(stop - start).count() / 1'000.0;

			total_nodes += alg.Nodes();
			total_duration += duration;

			Intensity print_intensity = { result.intensity.FullDepth() ? pos.EmptyCount() : result.intensity.depth, result.intensity.certainty };
			table.PrintRow(print_intensity, result.score, duration, alg.Nodes(), result.move);
		}
		table.PrintSeparator();
		table.PrintSummary(total_duration, total_nodes);
	}
};


class TestLab
{
	TestTable table;
public:
	void Run(std::ranges::range auto&& pos_scores, Intensity intensity, Algorithm& alg)
	{
		table.PrintHeader();
		std::vector<int> diff;
		diff.reserve(std::ranges::distance(pos_scores));
		uint64 total_nodes = 0;
		auto total_start = std::chrono::high_resolution_clock::now();
		for (auto&& [pos, score] : pos_scores)
		{
			auto start = std::chrono::high_resolution_clock::now();
			auto result = alg.Eval(pos, intensity);
			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = round<std::chrono::milliseconds>(stop - start).count() / 1'000.0;
			diff.push_back(result.score - score);
			total_nodes += alg.Nodes();
			Intensity print_intensity = { result.intensity.FullDepth() ? pos.EmptyCount() : result.intensity.depth, result.intensity.certainty };
			table.PrintRow(print_intensity, result.score, score, duration, alg.Nodes(), result.move);
		}
		auto total_stop = std::chrono::high_resolution_clock::now();
		auto total_duration = round<std::chrono::milliseconds>(total_stop - total_start).count() / 1'000.0;

		table.PrintSeparator();
		table.PrintSummary(StandardDeviation(diff), total_duration, total_nodes);
	}
};


void PrintHelp()
{
	std::cout
		<< "   -d <int[@float]>   Depth [and confidence]\n"
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
	
	Intensity intensity = Intensity::Exact();
	AAMSSE model;
	std::size_t buckets = 10'000'000;
	std::filesystem::path file;
	bool solve = false;
	bool test = false;

	for (int i = 0; i < argc; i++)
	{
		if (std::string(argv[i]) == "-d") intensity = ParseIntensity(argv[++i]);
		else if (std::string(argv[i]) == "-m") model = Deserialize<AAMSSE>(argv[++i]);
		else if (std::string(argv[i]) == "-tt") buckets = std::stoull(argv[++i]);
		else if (std::string(argv[i]) == "-solve") { file = argv[++i]; solve = true; }
		else if (std::string(argv[i]) == "-test") { file = argv[++i]; test = true; }
		else if (std::string(argv[i]) == "-h") { PrintHelp(); return 0; }
	}

	HT tt{ buckets };
	auto alg = IDAB<PVS>{ tt, model };
	
	if (solve)
	{
		std::vector<Position> pos;
		std::fstream stream(file, std::ios::in);
		for (std::string line; std::getline(stream, line); )
			pos.push_back(ParsePosition_SingleLine(line));

		SolverLab{}.Run(pos, intensity, alg);
	}
	else if (test)
	{
		std::vector<PosScore> ps;
		std::fstream stream(file, std::ios::in);
		for (std::string line; std::getline(stream, line); )
			ps.push_back(ParsePosScore_SingleLine(line));

		TestLab{}.Run(ps, intensity, alg);
	}
	else
	{
		try
		{
			Position pos = ParsePosition_SingleLine(std::string(argv[argc - 2]) + ' ' + argv[argc - 1]);
			auto start = std::chrono::high_resolution_clock::now();
			auto result = alg.Eval(pos, intensity);
			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = round<std::chrono::milliseconds>(stop - start).count() / 1'000.0;
			Intensity print_intensity = { result.intensity.FullDepth() ? pos.EmptyCount() : result.intensity.depth, result.intensity.certainty };
			SolverTable table;
			table.PrintRow(print_intensity, result.score, duration, alg.Nodes(), result.move);
		}
		catch (const std::exception& ex)
		{
			std::cerr << ex.what() << '\n';
		}
	}
	return 0;
}
