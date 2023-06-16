#include "IO/IO.h"
#include "Search/Search.h"
#include <cstdint>
#include <vector>
#include <iostream>

class RandomPositionGeneratorWithEmptyCount
{
	const int empty_count;
	std::mt19937_64 rnd_engine;
	std::uniform_int_distribution<int> boolean{ 0, 1 };
public:
	RandomPositionGeneratorWithEmptyCount(int empty_count, unsigned int seed = std::random_device{}())
		: empty_count(empty_count), rnd_engine(seed)
	{
		if (empty_count < 0 or empty_count > 64)
			throw std::runtime_error("'empty_count' out of bounds.");
	}

	Position operator()() noexcept
	{
		Position pos;
		while (pos.EmptyCount() > empty_count)
		{
			int rnd = std::uniform_int_distribution<int>(0, pos.EmptyCount() - 1)(rnd_engine);

			// deposit bit on an empty field
			uint64_t bit = PDep(1ULL << rnd, pos.Empties());

			if (boolean(rnd_engine))
				pos = Position{ pos.Player() | bit, pos.Opponent() };
			else
				pos = Position{ pos.Player(), pos.Opponent() | bit };
		}
		return pos;
	}
};

class TimingTable : public Table
{
public:
	TimingTable(std::string title)
		: Table(std::format("{:^22}|Empties|  TTS   |   Nodes/s   |    Nodes     ", title), "                      |{:>7}|{:>8}|{:>13L}|{:>14L}")
	{}

	void PrintRow(int empties, int sample_size, uint64_t nodes, std::chrono::duration<double> duration)
	{
		Table::PrintRow(
			empties,
			short_time_format(duration / sample_size),
			std::size_t(nodes / duration.count()),
			nodes,
			sample_size,
			duration
		);
	}
};

void TimeEndgame()
{
	//std::vector<std::vector<Position>> positions;
	//for (int e = 0; e <= 18; e++)
	//{
	//	std::vector<Position> p;
	//	p.reserve(100'000);
	//	RandomPositionGeneratorWithEmptyCount gen(e, /*seed*/ 13);
	//	for (int i = 0; i < 1'000; i++)
	//		p.push_back(gen());
	//	positions.push_back(std::move(p));
	//}

	//NegaMax nega_max;
	//TimingTable table("NegaMax");
	//table.PrintHeader();
	//for (int e = 0; e < 7; e++)
	//{
	//	uint64_t nodes = 0;
	//	auto start = std::chrono::high_resolution_clock::now();
	//	for (Position pos : positions[e])
	//	{
	//		nega_max.Eval(pos);
	//		nodes += nega_max.Nodes();
	//	}
	//	auto stop = std::chrono::high_resolution_clock::now();

	//	table.PrintRow(e, positions[e].size(), nodes, stop - start);
	//}
	//table.PrintSeparator();

	//AlphaBeta ab;
	//table = TimingTable("AlphaBeta");
	//table.PrintHeader();
	//for (int e = 0; e < 9; e++)
	//{
	//	uint64_t nodes = 0;
	//	auto start = std::chrono::high_resolution_clock::now();
	//	for (Position pos : positions[e])
	//	{
	//		ab.Eval(pos);
	//		nodes += ab.Nodes();
	//	}
	//	auto stop = std::chrono::high_resolution_clock::now();

	//	table.PrintRow(e, positions[e].size(), nodes, stop - start);
	//}
	//table.PrintSeparator();

	//HT tt{ 10'000'000 };
	//PVS pvs(tt);
	//TimingTable table = TimingTable("PVS");
	//table.PrintHeader();
	//for (int e = 0; e < positions.size(); e++)
	//{
	//	uint64_t nodes = 0;
	//	auto start = std::chrono::high_resolution_clock::now();
	//	for (Position pos : positions[e])
	//	{
	//		pvs.Eval(pos);
	//		nodes += pvs.Nodes();
	//	}
	//	auto stop = std::chrono::high_resolution_clock::now();

	//	table.PrintRow(e, positions[e].size(), nodes, stop - start);
	//}
	//table.PrintSeparator();


	std::vector<PosScore> data = LoadPosScoreFile("..\\data\\fforum-1-19.ps");
	PatternBasedEstimator evaluator = LoadPatternBasedEstimator("G:\\Reversi2\\iteration16.model");

	HT tt{ 10'000'000 };
	IDAB search(tt, evaluator);
	ResultTable table;
	table.PrintHeader();
	for (const PosScore& ps : data)
		table.PrintRow(search.Eval(ps.pos), ps.score);
	table.PrintSeparator();
	table.PrintSummary();
	std::cout << std::endl;

	data = LoadPosScoreFile("..\\data\\fforum-20-39.ps");
	tt.clear();
	table.clear();
	table.PrintHeader();
	for (const PosScore& ps : data)
		table.PrintRow(search.Eval(ps.pos), ps.score);
	table.PrintSeparator();
	table.PrintSummary();
}


int main(int argc, char* argv[])
{
	std::locale::global(std::locale(""));

	std::cout << "Endgame\n";
	TimeEndgame();

	return 0;
}

//void TimeMidgame()
//{
//	std::vector<Position> positions = RandomlyPlayedPositionsWithEmptyCount(100'000, 30);
//
//	std::vector<std::pair<std::string, std::unique_ptr<Algorithm>>> algs;
//
//	HT tt{ 10'000'000 };
//	AAGLEM model{ pattern::cassandra, 5 };
//	algs.emplace_back("PVS", std::make_unique<PVS>(tt, model));
//	algs.emplace_back("IDAB", std::make_unique<IDAB<PVS>>(tt, model));
//
//	TimingTable table("{:^4}|Depth|  TTS   |   Nodes/s   |    Nodes     ", "    |{:<5}|{:>8}|{:>13L}|{:>14L}");
//
//	for (auto&& [name, alg] : algs)
//	{
//		table.PrintHeader(name);
//		for (int d = 0; d <= 5; d++)
//		{
//			alg->clear();
//			uint64 nodes = 0;
//			auto start = std::chrono::high_resolution_clock::now();
//			for (Position pos : positions)
//			{
//				alg->Eval(pos, d);
//				nodes += alg->Nodes();
//			}
//			auto stop = std::chrono::high_resolution_clock::now();
//
//			table.PrintRow(d, positions.size(), nodes, stop - start);
//		}
//	}
//}
//
//int main(int argc, char* argv[])
//{
//	std::locale::global(std::locale(""));
//
//	fmt::print("Endgame\n");
//	TimeEndgame();
//
//	fmt::print("\nMidgame\n");
//	TimeMidgame();
//
//	return 0;
//}

//
//void Test(std::ranges::range auto&& puzzles, Algorithm& alg)
//{
//	std::chrono::duration<double> duration{ 0 };
//	uint64 nodes = 0;
//	SolverTable table(/*test*/ true);
//	table.PrintHeader();
//
//	std::vector<int> diff;
//	Process(std::execution::seq, puzzles,
//		[&](PosScore& p, std::size_t index) {
//			int old_score = p.score;
//			auto start = std::chrono::high_resolution_clock::now();
//			int new_score = alg.Eval(p.pos);
//			auto stop = std::chrono::high_resolution_clock::now();
//			duration += stop - start;
//			nodes += alg.Nodes();
//			diff.push_back(old_score - new_score);
//			table.PrintRow(index, p.pos.EmptyCount(), old_score, new_score, stop - start, alg.Nodes());
//		});
//	table.PrintSeparator();
//	table.PrintSummary(StandardDeviation(diff), duration, nodes);
//}
//void Test(std::ranges::range auto&& puzzles, Algorithm&& alg) { Test (puzzles, alg); }
//
//DurationNodes Time(auto execution_policy, Algorithm& alg, Intensity intensity, std::ranges::random_access_range auto&& pos)
//{
//	uint64 nodes = 0;
//	double duration{ 0 };
//	std::size_t size = std::size(pos);
//	#pragma omp parallel reduction(+:nodes) reduction(+:duration) if (std::is_same_v<decltype(execution_policy), std::execution::parallel_policy>)
//	{
//		auto start = std::chrono::high_resolution_clock::now();
//		#pragma omp for nowait
//		for (int i = 0; i < size; i++)
//		{
//			alg.Eval(pos[i], intensity);
//			nodes += alg.Nodes();
//		}
//		auto stop = std::chrono::high_resolution_clock::now();
//		duration += (stop - start).count();
//	}
//	if (std::is_same_v<decltype(execution_policy), std::execution::parallel_policy>)
//		duration /= std::thread::hardware_concurrency();
//	return { nodes, std::chrono::duration<double, std::nano>(duration) };
//}
//
//void TimeMidgame(const AAGLEM& model, std::ranges::range auto&& pos)
//{
//	int samples = 200;
//	auto e20 = pos | FilterByEmptyCount(20) | ranges::views::take(samples);
//	auto e30 = pos | FilterByEmptyCount(30) | ranges::views::take(samples);
//	auto e40 = pos | FilterByEmptyCount(40) | ranges::views::take(samples);
//	HT tt(10'000'000);
//	PVS pvs{ tt, model };
//	IDAB<PVS> idab{ tt, model };
//
//	MidgameTable table;
//	table.PrintHeader();
//	for (Confidence certainty : { Confidence::Certain(), 1.1_sigmas })
//	{
//		int D = certainty.IsCertain() ? 10 : 14;
//		for (int d = 0; d <= D; d++)
//		{
//			Intensity intensity{ d, certainty };
//			pvs.clear();
//			auto t1 = Time(std::execution::seq, pvs, intensity, e20).duration / samples;
//			auto t2 = Time(std::execution::seq, pvs, intensity, e30).duration / samples;
//			auto t3 = Time(std::execution::seq, pvs, intensity, e40).duration / samples;
//			idab.clear();
//			auto t4 = Time(std::execution::seq, idab, intensity, e20).duration / samples;
//			auto t5 = Time(std::execution::seq, idab, intensity, e30).duration / samples;
//			auto t6 = Time(std::execution::seq, idab, intensity, e40).duration / samples;
//			table.PrintRow(intensity, t1, t2, t3, t4, t5, t6);
//		}
//		table.PrintSeparator();
//	}
//}
//
//void TimeEndgame(const AAGLEM& model, std::ranges::range auto&& pos, std::vector<int> samples)
//{
//	HT tt(10'000'000);
//	PVS pvs{ tt, model };
//	IDAB<PVS> idab{ tt, model };
//
//	EndgameTable table;
//	table.PrintHeader();
//	for (Confidence certainty : { Confidence::Certain(), 1.1_sigmas })
//	{
//		for (int e = 11; e < samples.size(); e++)
//		{
//			auto data = pos | FilterByEmptyCount(e) | ranges::views::take(samples[e]);
//			Intensity intensity{ e, certainty };
//			pvs.clear();
//			auto t1 = Time(std::execution::seq, pvs, intensity, data).duration / samples[e];
//			pvs.clear();
//			auto t2 = Time(std::execution::seq, idab, intensity, data).duration / samples[e];
//			table.PrintRow(intensity, t1, t2);
//		}
//		table.PrintSeparator();
//	}
//}
//
//void TimeEndgame(std::string_view name, Algorithm& alg, const std::vector<Position>& pos)
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
//
//void AllPos(int e)
//{
//	std::string file = fmt::format(R"(G:\Reversi\e{}_all.script)", e);
//	std::fstream stream(file, std::ios::out);
//	for (Position pos : UniqueChildren(Position::Start(), e))
//		stream << SingleLine(pos) << '\n';
//}

//int main(int argc, char* argv[])
//{
//	//std::string file = R"(G:\Reversi\e50_1k.script)";
//	//std::fstream stream(file, std::ios::out);
//	//for (Position pos : PosGen::generate_n_unique(1'000, PosGen::RandomlyPlayed(/*seed*/ 12, /*seed*/ 13, 50)))
//	//	stream << SingleLine(pos) << '\n';
//	//return 0;
//	std::cout.imbue(std::locale(""));
//	RandomlyPlayedPositionsWithEmptyCount();
//
//	HT tt(10'000'000);
//	AAGLEM model = Deserialize<AAGLEM>(R"(G:\Reversi\d5_d5_model_9.mdl)");
//
//	Test(FForum_1, IDAB<PVS>{ tt, model }); std::cout << '\n';
//	Test(FForum_2, IDAB<PVS>{ tt, model }); std::cout << '\n';
//	Test(FForum_3, IDAB<PVS>{ tt, model }); std::cout << '\n';
//	Test(FForum_4, IDAB<PVS>{ tt, model }); std::cout << '\n';
//	////return 0;
//
//	//for (int i = 2; i <= 40; i++) {
//	//	HT tt(10'000'000);
//	//	AAGLEM model = Deserialize<AAGLEM>(fmt::format(R"(G:\Reversi\d5_d5_model_{}.mdl)", i));
//	//	auto alg = IDAB<PVS>{ tt, model };
//	//	//Test(FForum_1, alg); std::cout << '\n';
//	//	//Test(FForum_2, alg); std::cout << '\n';
//	//	Test(FForum_3, alg); std::cout << '\n';
//	//}
//	////std::cout << PVS::counter << '\n';
//	//return 0;
//
//	//DB<Position> db;
//	//for (int e = 0; e <= 50; e++)
//	//	db.Add({}, PosGen::generate_n_unique(100'000, PosGen::RandomlyPlayed(/*seed*/ 12, /*seed*/ 13, e)));
//
//	//std::vector<int> samples_negamax = {
//	//	100'000, // 0
//	//	100'000, // 1
//	//	100'000, // 2
//	//	100'000, // 3
//	//	100'000, // 4
//	//	100'000, // 5
//	//	100'000, // 6
//	//	100'000, // 7
//	//	 50'000, // 8
//	//	 10'000, // 9
//	//	  2'000, // 10
//	//	    400, // 11
//	//};
//
//	//std::vector<int> samples_alphabeta = {
//	//	100'000, // 0
//	//	100'000, // 1
//	//	100'000, // 2
//	//	100'000, // 3
//	//	100'000, // 4
//	//	100'000, // 5
//	//	100'000, // 6
//	//	100'000, // 7
//	//	100'000, // 8
//	//	 60'000, // 9
//	//	 20'000, // 10
//	//	  7'000, // 11
//	//	  3'000, // 12
//	//	  1'000, // 13
//	//		400, // 14
//	//};
//
//	//std::vector<int> samples_pvs = {
//	//	100'000, // 0
//	//	100'000, // 1
//	//	100'000, // 2
//	//	100'000, // 3
//	//	100'000, // 4
//	//	100'000, // 5
//	//	100'000, // 6
//	//	100'000, // 7
//	//	100'000, // 8
//	//	100'000, // 9
//	//	 60'000, // 10
//	//	 20'000, // 11
//	//	 10'000, // 12
//	//	  3'000, // 13
//	//      1'500, // 14
//	//	    500, // 15
//	//		200, // 16
//	//		200, // 17
//	//		200, // 18
//	//		200, // 19
//	//		200, // 20
//	//};
//
//	//TimeEndgame(std::execution::seq, "PVS", PVS{ tt, model }, db, samples_pvs);
//	//tt.clear();
//	//TimeEndgame(std::execution::seq, "IDAB", IDAB<PVS>{ tt, model }, db, samples_pvs);
//
//	//Time(std::execution::seq, "NegaMax seq", NegaMax{}, db, samples_negamax_seq);
//	//Time(std::execution::par, "NegaMax par", NegaMax{}, db, samples_negamax_par);
//
//	//Time(std::execution::seq, "AlphaBetaFailHard seq", AlphaBetaFailHard{}, db, samples_alphabeta_seq);
//	//Time(std::execution::par, "AlphaBetaFailHard par", AlphaBetaFailHard{}, db, samples_alphabeta_par);
//
//	//TimeEndgame(model, db, samples_pvs);
//	//TimeMidgame(model, db);
//
//	//Time(std::execution::par, "PVS par", PVS{tt}, db, samples_negamax_par);
//
//	//TimeEndgame(std::execution::seq, "AlphaBetaFailSoft seq", AlphaBetaFailSoft{}, db, samples_alphabeta_seq);
//	//Time(std::execution::par, "AlphaBetaFailSoft par", AlphaBetaFailSoft{}, db, samples_alphabeta_par);
//
//	//TimeEndgame(std::execution::seq, "AlphaBetaFailSuperSoft seq", AlphaBetaFailSuperSoft{}, db, samples_alphabeta);
//	//Time(std::execution::par, "AlphaBetaFailSuperSoft par", AlphaBetaFailSuperSoft{}, db, samples_alphabeta_par);
//
//	//Time(std::execution::seq, "AlphaBetaFailHard seq", MTD<0, AlphaBetaFailHard>{}, db, samples_alphabeta_seq);
//	//Time(std::execution::par, "AlphaBetaFailHard par", MTD<0, AlphaBetaFailHard>{}, db, samples_alphabeta_par);
//
//	//Time(std::execution::seq, "AlphaBetaFailSoft seq", MTD<0, AlphaBetaFailSoft>{}, db, samples_alphabeta_seq);
//	//Time(std::execution::par, "AlphaBetaFailSoft par", MTD<0, AlphaBetaFailSoft>{}, db, samples_alphabeta_par);
//
//	//Time(std::execution::seq, "AlphaBetaFailSuperSoft seq", MTD<0, AlphaBetaFailSuperSoft>{}, db, samples_alphabeta_seq);
//	//Time(std::execution::par, "AlphaBetaFailSuperSoft par", MTD<0, AlphaBetaFailSuperSoft>{}, db, samples_alphabeta_par);
//
//	//std::cout
//	//	<< "\nEndgame\n"
//	//	<< " Empties |  depth |     TTS     |       Speed       |       Nodes     | Duration \n"
//	//	<< "---------+--------+-------------+-------------------+-----------------+----------\n";
//	//for (int E = 0; E <= 25; E++)
//	//{
//	//	Request request(E);
//	//	std::vector<Puzzle> to_bench;
//	//	for (int i = 0; i < benchmark.size(); i++)
//	//	{
//	//		Puzzle& p = benchmark[i];
//	//		if (p.EmptyCount() == E and i % (E * E / 10 + 1) == 0)
//	//		{
//	//			to_bench.push_back(p);
//	//			to_bench.back().clear();
//	//			to_bench.back().insert(request);
//	//		}
//	//	}
//	//	Benchmark(std::execution::par, E, request, to_bench);
//	//}
//
//	//DB<NoMovePuzzle> benchmark;
//	//for (int i = 0; i < 10; i++)
//	//	for (int j = 0; j < 10; j++)
//	//	{
//	//		try
//	//		{
//	//			benchmark.Add(fmt::format(R"(G:\Reversi\play{}{}_benchmark.puz)", i, j));
//	//		}
//	//		catch (...) {}
//	//	}
//
//	//std::cout
//	//	<< "Midgame\n"
//	//	<< " Empties |  depth |     TTS     |       Speed       |       Nodes     | Duration \n"
//	//	<< "---------+--------+-------------+-------------------+-----------------+----------\n";
//	//for (int d = 0; d <= 10; d++)
//	//{
//	//	Request request(d);
//	//	std::vector<Puzzle> to_bench;
//	//	for (int i = 0; i < benchmark.size(); i++)
//	//	{
//	//		Puzzle& p = benchmark[i];
//	//		if (p.EmptyCount() > d and i % (d * d + 1) == 0)
//	//		{
//	//			to_bench.push_back(p);
//	//			to_bench.back().clear();
//	//			to_bench.back().push_back(request);
//	//		}
//	//	}
//	//	Benchmark(std::execution::par, "", request, to_bench);
//	//}
//
//	//std::cout
//	//	<< "\nSelective endgame\n"
//	//	<< " Empties |  depth |     TTS     |       Speed       |       Nodes     | Duration \n"
//	//	<< "---------+--------+-------------+-------------------+-----------------+----------\n";
//	//for (int E = 0; E <= 25; E++)
//	//{
//	//	Request request(E, 1.1_sigmas);
//	//	std::vector<Puzzle> to_bench;
//	//	for (int i = 0; i < benchmark.size(); i++)
//	//	{
//	//		Puzzle& p = benchmark[i];
//	//		if (p.EmptyCount() == E and i % (E + 1) == 0)
//	//		{
//	//			to_bench.push_back(p);
//	//			to_bench.back().clear();
//	//			to_bench.back().push_back(request);
//	//		}
//	//	}
//	//	Benchmark(std::execution::par, E, request, to_bench);
//	//}
//
//	//return 0;
//
//	/*std::cout.imbue(std::locale(""));
//	TimePatternEval();
//	TimeLateEndgame();
//
//	int random_seed_1 = 4182;
//	int random_seed_2 = 5899;
//	int size = 100;
//	AAGLEM pattern_eval = DefaultPatternEval();
//	HashTablePVS tt{ 1'000'000 };
//
//	std::cout
//		<< " Empties | depth |     TTS     |      Speed      |      Nodes     | Duration \n"
//		<< "---------+-------+-------------+-----------------+----------------+----------\n";
//
//	std::vector<std::vector<Puzzle>> groups;
//
//	std::vector<int> depth_sizes = { 10'000, 10'000, 1'000, 1'000, 1'000, 100, 1'000, 100, 100, 100, 100 };
//	for (int d = 0; d <= 10; d++)
//	{
//		std::vector<Puzzle> group;
//		group.reserve(depth_sizes[d] * 31);
//		for (int e = 20; e <= 50; e += 10)
//		{
//			for (const Position& pos : generate_n_unique(PosGen::RandomlyPlayed(random_seed_1 + d, random_seed_2 + d, e), depth_sizes[d]))
//				group.push_back(Puzzle{ pos, Puzzle::Task{ Request(d) } });
//		}
//		groups.push_back(std::move(group));
//	}
//	for (int e = 15; e <= 30; e++)
//	{
//		std::vector<Puzzle> group;
//		group.reserve(size);
//		for (const Position& pos : generate_n_unique(PosGen::RandomlyPlayed(random_seed_1 + e, random_seed_2 + e, e), size))
//			group.push_back(Puzzle{ pos, Puzzle::Task{ Request::ExactScore(pos) } });
//		groups.push_back(std::move(group));
//	}
//
//	auto ex = CreateExecutor(std::execution::seq, groups,
//		[&](Puzzle& p) { p.Solve(IDAB{ tt, pattern_eval }); });
//
//	Metronome reporter(1s, [&]() {
//		static std::size_t index = 0;
//		while (index < groups.size() and ex->IsDone(index)) {
//			auto duration = Duration(groups[index]);
//			auto nodes = Nodes(groups[index]);
//			auto e = groups[index].front().EmptyCount();
//			auto d = groups[index].front().tasks.front().GetIntensity().depth;
//			std::cout
//				<< fmt::format(std::locale(""), "{:8} | {:5} | {}/pos | {:11L} N/s | {:14L} | {:0.2f} s\n",
//					e, d, short_time_format(duration / size), std::size_t(nodes / duration.count()), nodes, duration.count());
//			index++;
//		}
//	});
//
//	reporter.Start();
//	ex->Process();
//	reporter.Force();
//
//	return 0;*/
//}