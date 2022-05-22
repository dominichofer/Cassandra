#include "Core/Core.h"
#include "IO/IO.h"
#include "Math/Math.h"
#include "Pattern/Pattern.h"
#include "PatternIO/PatternIO.h"
#include "Search/Search.h"
#include <vector>
#include <iostream>
#include <execution>

using namespace std::chrono_literals;

class MidgameTable : public Table
{
public:
	MidgameTable()
	{
		AddColumn("e20", 8, "{:>8}");
		AddColumn("e30", 8, "{:>8}");
		AddColumn("e40", 8, "{:>8}");
		AddColumn("Depth", 6, "{:>6}");
		AddColumn("e20", 8, "{:>8}");
		AddColumn("e30", 8, "{:>8}");
		AddColumn("e40", 8, "{:>8}");
	}

	void PrintRow(
		Intensity intensity,
		std::chrono::duration<double> t1,
		std::chrono::duration<double> t2,
		std::chrono::duration<double> t3,
		std::chrono::duration<double> t4,
		std::chrono::duration<double> t5,
		std::chrono::duration<double> t6)
	{
		Table::PrintRow(
			short_time_format(t1),
			short_time_format(t2),
			short_time_format(t3),
			intensity,
			short_time_format(t4),
			short_time_format(t5),
			short_time_format(t6)
		);
	}
};

struct DurationNodes
{
	uint64 nodes;
	std::chrono::duration<double> duration;
};

class EndgameTable : public Table
{
	std::string name;
public:
	EndgameTable()
	{
		AddColumn("PVS", 8, "{:>8}");
		AddColumn("Depth", 6, "{:>6}");
		AddColumn("IDAB", 8, "{:>8}");
	}

	void PrintRow(
		Intensity intensity,
		std::chrono::duration<double> t1,
		std::chrono::duration<double> t2)
	{
		Table::PrintRow(
			short_time_format(t1),
			intensity,
			short_time_format(t2)
		);
	}
};

//class EndgameTable : public Table
//{
//	std::string name;
//public:
//	EndgameTable()
//	{
//		AddColumn("", 9, "{:<9}");
//		AddColumn("Empties", 7, "{:>7}");
//		AddColumn("TTS", 8, "{:>8}");
//		AddColumn("Nodes/s", 13, "{:>13L}");
//		AddColumn("Nodes", 14, "{:>14L}");
//		AddColumn("Samples", 9, "{:>9L}");
//		AddColumn("Duration", 9, "{:>9.2}");
//	}
//
//	void NextName(const std::string& name)
//	{
//		this->name = name;
//	}
//
//	void PrintRow(int empty_count, int samples, DurationNodes x)
//	{
//		Table::PrintRow(
//			name,
//			empty_count,
//			short_time_format(x.duration / samples),
//			std::size_t(x.nodes / x.duration.count()),
//			x.nodes,
//			samples,
//			x.duration
//		);
//		name = "";
//	}
//};

class SolverTable : public Table
{
	bool test;
public:
	SolverTable(bool test) : test(test)
	{
		AddColumn("#", 2, "{:>2L}");
		AddColumn("depth", 6, "{:6}");
		AddColumn("eval", 5, " {:+03} ");
		AddColumn("score", 5, " {:+03} ", test);
		AddColumn("diff", 5, " {:+03} ", test);
		AddColumn("time [s]", 16, "{:>#16.3f}");
		AddColumn("nodes", 18, "{:>18L}");
		AddColumn("nodes/s", 14, "{:>14.0Lf}");
	}
	void PrintRow(std::size_t index, const Intensity& intensity, int score, int result, std::chrono::duration<double> duration, uint64 nodes)
	{
		int diff = score - result;

		Table::PrintRow(
			index,
			to_string(intensity),
			score,
			result,
			diff ? std::optional(diff) : std::nullopt,
			duration.count(),
			nodes,
			duration.count() ? std::optional(nodes / duration.count()) : std::nullopt
		);
	}
	void PrintSummary(double std_dev, auto duration, auto nodes)
	{
		Table::PrintRow(
			Table::Empty,
			Table::Empty,
			Table::Empty,
			Table::Empty,
			std_dev ? std::optional(std_dev) : std::nullopt,
			duration.count(),
			nodes,
			nodes / duration.count()
		);
	}
};

void Test(std::ranges::range auto&& puzzles, Algorithm& alg)
{
	std::chrono::duration<double> duration{ 0 };
	uint64 nodes = 0;
	SolverTable table(/*test*/ true);
	table.PrintHeader();

	std::vector<int> diff;
	Process(std::execution::seq, puzzles,
		[&](PosScore& p, std::size_t index) {
			int old_score = p.score;
			auto start = std::chrono::high_resolution_clock::now();
			int new_score = alg.Eval(p.pos);
			auto stop = std::chrono::high_resolution_clock::now();
			duration += stop - start;
			nodes += alg.Nodes();
			diff.push_back(old_score - new_score);
			table.PrintRow(index, p.pos.EmptyCount(), old_score, new_score, stop - start, alg.Nodes());
		});
	table.PrintSeparator();
	table.PrintSummary(StandardDeviation(diff), duration, nodes);
}
void Test(std::ranges::range auto&& puzzles, Algorithm&& alg) { Test (puzzles, alg); }

DurationNodes Time(auto execution_policy, Algorithm& alg, Intensity intensity, std::ranges::random_access_range auto&& pos)
{
	uint64 nodes = 0;
	double duration{ 0 };
	std::size_t size = std::size(pos);
	#pragma omp parallel reduction(+:nodes) reduction(+:duration) if (std::is_same_v<decltype(execution_policy), std::execution::parallel_policy>)
	{
		auto start = std::chrono::high_resolution_clock::now();
		#pragma omp for nowait
		for (int i = 0; i < size; i++)
		{
			alg.Eval(pos[i], intensity);
			nodes += alg.Nodes();
		}
		auto stop = std::chrono::high_resolution_clock::now();
		duration += (stop - start).count();
	}
	if (std::is_same_v<decltype(execution_policy), std::execution::parallel_policy>)
		duration /= std::thread::hardware_concurrency();
	return { nodes, std::chrono::duration<double, std::nano>(duration) };
}
DurationNodes Time(auto execution_policy, Algorithm& alg, Intensity intensity, std::ranges::range auto&& pos)
{
	return Time(execution_policy, alg, intensity, pos | ranges::to_vector);
}

int EmptyCount(std::convertible_to<Position> auto& pos)
{
	return static_cast<Position>(pos).EmptyCount();
}

auto FilterByEmptyCount(int empty_count) { return ranges::views::filter([empty_count](const auto& p) { return EmptyCount(p) == empty_count; }); }
auto FilterByEmptyCount(int min, int max) { return ranges::views::filter([min, max](const auto& p) { auto e = EmptyCount(p); return min <= e and e <= max; }); }

void TimeMidgame(const AAGLEM& model, std::ranges::range auto&& pos)
{
	int samples = 200;
	auto e20 = pos | FilterByEmptyCount(20) | ranges::views::take(samples);
	auto e30 = pos | FilterByEmptyCount(30) | ranges::views::take(samples);
	auto e40 = pos | FilterByEmptyCount(40) | ranges::views::take(samples);
	HT tt(10'000'000);
	PVS pvs{ tt, model };
	IDAB<PVS> idab{ tt, model };

	MidgameTable table;
	fmt::print("              PVS               |        |               IDAB             \n");
	fmt::print("--------------------------------+--------+--------------------------------\n");
	table.PrintHeader();
	for (Confidence certainty : { Confidence::Certain(), 1.1_sigmas })
	{
		int D = certainty.IsCertain() ? 10 : 14;
		for (int d = 0; d <= D; d++)
		{
			Intensity intensity{ d, certainty };
			pvs.clear();
			auto t1 = Time(std::execution::seq, pvs, intensity, e20).duration / samples;
			auto t2 = Time(std::execution::seq, pvs, intensity, e30).duration / samples;
			auto t3 = Time(std::execution::seq, pvs, intensity, e40).duration / samples;
			idab.clear();
			auto t4 = Time(std::execution::seq, idab, intensity, e20).duration / samples;
			auto t5 = Time(std::execution::seq, idab, intensity, e30).duration / samples;
			auto t6 = Time(std::execution::seq, idab, intensity, e40).duration / samples;
			table.PrintRow(intensity, t1, t2, t3, t4, t5, t6);
		}
		table.PrintSeparator();
	}
}

void TimeEndgame(const AAGLEM& model, std::ranges::range auto&& pos, std::vector<int> samples)
{
	HT tt(10'000'000);
	PVS pvs{ tt, model };
	IDAB<PVS> idab{ tt, model };

	EndgameTable table;
	table.PrintHeader();
	for (Confidence certainty : { Confidence::Certain(), 1.1_sigmas })
	{
		for (int e = 11; e < samples.size(); e++)
		{
			auto data = pos | FilterByEmptyCount(e) | ranges::views::take(samples[e]);
			Intensity intensity{ e, certainty };
			pvs.clear();
			auto t1 = Time(std::execution::seq, pvs, intensity, data).duration / samples[e];
			pvs.clear();
			auto t2 = Time(std::execution::seq, idab, intensity, data).duration / samples[e];
			table.PrintRow(intensity, t1, t2);
		}
		table.PrintSeparator();
	}
}

//void TimeEndgame(auto execution_policy, const std::string& name, Algorithm& alg, const std::ranges::range auto& pos, std::vector<int> samples)
//{
//	EndgameTable table;
//	table.PrintHeader();
//	table.NextName(name);
//	for (int e = 0; e < samples.size(); e++)
//	{
//		auto data = pos | FilterByEmptyCount(e) | ranges::views::take(samples[e]);
//		auto ret = Time(execution_policy, alg, e, data);
//		table.PrintRow(e, samples[e], ret);
//	}
//	table.PrintSeparator();
//}
//void TimeEndgame(auto execution_policy, const std::string& name, Algorithm&& alg, DB<Position>& db, std::vector<int> samples)
//{
//	TimeEndgame(execution_policy, name, alg, db, samples);
//}

void AllPos(int e)
{
	std::string file = fmt::format(R"(G:\Reversi\e{}_all.script)", e);
	std::fstream stream(file, std::ios::out);
	for (Position pos : AllUnique(Position::Start(), e))
		stream << SingleLine(pos) << '\n';
}

int main(int argc, char* argv[])
{
	//std::string file = R"(G:\Reversi\e50_1k.script)";
	//std::fstream stream(file, std::ios::out);
	//for (Position pos : PosGen::generate_n_unique(1'000, PosGen::RandomlyPlayed(/*seed*/ 12, /*seed*/ 13, 50)))
	//	stream << SingleLine(pos) << '\n';
	//return 0;
	std::cout.imbue(std::locale(""));

	HT tt(10'000'000);
	AAGLEM model = Deserialize<AAGLEM>(R"(G:\Reversi\d5_d5_model_9.mdl)");

	Test(FForum_1, IDAB<PVS>{ tt, model }); std::cout << '\n';
	Test(FForum_2, IDAB<PVS>{ tt, model }); std::cout << '\n';
	Test(FForum_3, IDAB<PVS>{ tt, model }); std::cout << '\n';
	Test(FForum_4, IDAB<PVS>{ tt, model }); std::cout << '\n';
	////return 0;

	//for (int i = 2; i <= 40; i++) {
	//	HT tt(10'000'000);
	//	AAGLEM model = Deserialize<AAGLEM>(fmt::format(R"(G:\Reversi\d5_d5_model_{}.mdl)", i));
	//	auto alg = IDAB<PVS>{ tt, model };
	//	//Test(FForum_1, alg); std::cout << '\n';
	//	//Test(FForum_2, alg); std::cout << '\n';
	//	Test(FForum_3, alg); std::cout << '\n';
	//}
	////std::cout << PVS::counter << '\n';
	//return 0;

	DB<Position> db;
	for (int e = 0; e <= 50; e++)
		db.Add({}, PosGen::generate_n_unique(100'000, PosGen::RandomlyPlayed(/*seed*/ 12, /*seed*/ 13, e)));

	std::vector<int> samples_negamax = {
		100'000, // 0
		100'000, // 1
		100'000, // 2
		100'000, // 3
		100'000, // 4
		100'000, // 5
		100'000, // 6
		100'000, // 7
		 50'000, // 8
		 10'000, // 9
		  2'000, // 10
		    400, // 11
	};

	std::vector<int> samples_alphabeta = {
		100'000, // 0
		100'000, // 1
		100'000, // 2
		100'000, // 3
		100'000, // 4
		100'000, // 5
		100'000, // 6
		100'000, // 7
		100'000, // 8
		 60'000, // 9
		 20'000, // 10
		  7'000, // 11
		  3'000, // 12
		  1'000, // 13
			400, // 14
	};

	std::vector<int> samples_pvs = {
		100'000, // 0
		100'000, // 1
		100'000, // 2
		100'000, // 3
		100'000, // 4
		100'000, // 5
		100'000, // 6
		100'000, // 7
		100'000, // 8
		100'000, // 9
		 60'000, // 10
		 20'000, // 11
		 10'000, // 12
		  3'000, // 13
	      1'500, // 14
		    500, // 15
			200, // 16
			200, // 17
			200, // 18
			200, // 19
			200, // 20
	};

	//TimeEndgame(std::execution::seq, "PVS", PVS{ tt, model }, db, samples_pvs);
	//tt.clear();
	//TimeEndgame(std::execution::seq, "IDAB", IDAB<PVS>{ tt, model }, db, samples_pvs);

	//Time(std::execution::seq, "NegaMax seq", NegaMax{}, db, samples_negamax_seq);
	//Time(std::execution::par, "NegaMax par", NegaMax{}, db, samples_negamax_par);

	//Time(std::execution::seq, "AlphaBetaFailHard seq", AlphaBetaFailHard{}, db, samples_alphabeta_seq);
	//Time(std::execution::par, "AlphaBetaFailHard par", AlphaBetaFailHard{}, db, samples_alphabeta_par);

	TimeEndgame(model, db, samples_pvs);
	TimeMidgame(model, db);

	//Time(std::execution::par, "PVS par", PVS{tt}, db, samples_negamax_par);

	//TimeEndgame(std::execution::seq, "AlphaBetaFailSoft seq", AlphaBetaFailSoft{}, db, samples_alphabeta_seq);
	//Time(std::execution::par, "AlphaBetaFailSoft par", AlphaBetaFailSoft{}, db, samples_alphabeta_par);

	//TimeEndgame(std::execution::seq, "AlphaBetaFailSuperSoft seq", AlphaBetaFailSuperSoft{}, db, samples_alphabeta);
	//Time(std::execution::par, "AlphaBetaFailSuperSoft par", AlphaBetaFailSuperSoft{}, db, samples_alphabeta_par);

	//Time(std::execution::seq, "AlphaBetaFailHard seq", MTD<0, AlphaBetaFailHard>{}, db, samples_alphabeta_seq);
	//Time(std::execution::par, "AlphaBetaFailHard par", MTD<0, AlphaBetaFailHard>{}, db, samples_alphabeta_par);

	//Time(std::execution::seq, "AlphaBetaFailSoft seq", MTD<0, AlphaBetaFailSoft>{}, db, samples_alphabeta_seq);
	//Time(std::execution::par, "AlphaBetaFailSoft par", MTD<0, AlphaBetaFailSoft>{}, db, samples_alphabeta_par);

	//Time(std::execution::seq, "AlphaBetaFailSuperSoft seq", MTD<0, AlphaBetaFailSuperSoft>{}, db, samples_alphabeta_seq);
	//Time(std::execution::par, "AlphaBetaFailSuperSoft par", MTD<0, AlphaBetaFailSuperSoft>{}, db, samples_alphabeta_par);

	//std::cout
	//	<< "\nEndgame\n"
	//	<< " Empties |  depth |     TTS     |       Speed       |       Nodes     | Duration \n"
	//	<< "---------+--------+-------------+-------------------+-----------------+----------\n";
	//for (int E = 0; E <= 25; E++)
	//{
	//	Request request(E);
	//	std::vector<Puzzle> to_bench;
	//	for (int i = 0; i < benchmark.size(); i++)
	//	{
	//		Puzzle& p = benchmark[i];
	//		if (p.EmptyCount() == E and i % (E * E / 10 + 1) == 0)
	//		{
	//			to_bench.push_back(p);
	//			to_bench.back().clear();
	//			to_bench.back().insert(request);
	//		}
	//	}
	//	Benchmark(std::execution::par, E, request, to_bench);
	//}

	//DB<NoMovePuzzle> benchmark;
	//for (int i = 0; i < 10; i++)
	//	for (int j = 0; j < 10; j++)
	//	{
	//		try
	//		{
	//			benchmark.Add(fmt::format(R"(G:\Reversi\play{}{}_benchmark.puz)", i, j));
	//		}
	//		catch (...) {}
	//	}

	//std::cout
	//	<< "Midgame\n"
	//	<< " Empties |  depth |     TTS     |       Speed       |       Nodes     | Duration \n"
	//	<< "---------+--------+-------------+-------------------+-----------------+----------\n";
	//for (int d = 0; d <= 10; d++)
	//{
	//	Request request(d);
	//	std::vector<Puzzle> to_bench;
	//	for (int i = 0; i < benchmark.size(); i++)
	//	{
	//		Puzzle& p = benchmark[i];
	//		if (p.EmptyCount() > d and i % (d * d + 1) == 0)
	//		{
	//			to_bench.push_back(p);
	//			to_bench.back().clear();
	//			to_bench.back().push_back(request);
	//		}
	//	}
	//	Benchmark(std::execution::par, "", request, to_bench);
	//}

	//std::cout
	//	<< "\nSelective endgame\n"
	//	<< " Empties |  depth |     TTS     |       Speed       |       Nodes     | Duration \n"
	//	<< "---------+--------+-------------+-------------------+-----------------+----------\n";
	//for (int E = 0; E <= 25; E++)
	//{
	//	Request request(E, 1.1_sigmas);
	//	std::vector<Puzzle> to_bench;
	//	for (int i = 0; i < benchmark.size(); i++)
	//	{
	//		Puzzle& p = benchmark[i];
	//		if (p.EmptyCount() == E and i % (E + 1) == 0)
	//		{
	//			to_bench.push_back(p);
	//			to_bench.back().clear();
	//			to_bench.back().push_back(request);
	//		}
	//	}
	//	Benchmark(std::execution::par, E, request, to_bench);
	//}

	//return 0;

	/*std::cout.imbue(std::locale(""));
	TimePatternEval();
	TimeLateEndgame();

	int random_seed_1 = 4182;
	int random_seed_2 = 5899;
	int size = 100;
	AAGLEM pattern_eval = DefaultPatternEval();
	HashTablePVS tt{ 1'000'000 };

	std::cout
		<< " Empties | depth |     TTS     |      Speed      |      Nodes     | Duration \n"
		<< "---------+-------+-------------+-----------------+----------------+----------\n";

	std::vector<std::vector<Puzzle>> groups;

	std::vector<int> depth_sizes = { 10'000, 10'000, 1'000, 1'000, 1'000, 100, 1'000, 100, 100, 100, 100 };
	for (int d = 0; d <= 10; d++)
	{
		std::vector<Puzzle> group;
		group.reserve(depth_sizes[d] * 31);
		for (int e = 20; e <= 50; e += 10)
		{
			for (const Position& pos : generate_n_unique(PosGen::RandomlyPlayed(random_seed_1 + d, random_seed_2 + d, e), depth_sizes[d]))
				group.push_back(Puzzle{ pos, Puzzle::Task{ Request(d) } });
		}
		groups.push_back(std::move(group));
	}
	for (int e = 15; e <= 30; e++)
	{
		std::vector<Puzzle> group;
		group.reserve(size);
		for (const Position& pos : generate_n_unique(PosGen::RandomlyPlayed(random_seed_1 + e, random_seed_2 + e, e), size))
			group.push_back(Puzzle{ pos, Puzzle::Task{ Request::ExactScore(pos) } });
		groups.push_back(std::move(group));
	}

	auto ex = CreateExecutor(std::execution::seq, groups,
		[&](Puzzle& p) { p.Solve(IDAB{ tt, pattern_eval }); });

	Metronome reporter(1s, [&]() {
		static std::size_t index = 0;
		while (index < groups.size() and ex->IsDone(index)) {
			auto duration = Duration(groups[index]);
			auto nodes = Nodes(groups[index]);
			auto e = groups[index].front().EmptyCount();
			auto d = groups[index].front().tasks.front().GetIntensity().depth;
			std::cout
				<< fmt::format(std::locale(""), "{:8} | {:5} | {}/pos | {:11L} N/s | {:14L} | {:0.2f} s\n",
					e, d, short_time_format(duration / size), std::size_t(nodes / duration.count()), nodes, duration.count());
			index++;
		}
	});

	reporter.Start();
	ex->Process();
	reporter.Force();

	return 0;*/
}