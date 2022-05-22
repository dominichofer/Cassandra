#include "Core/Core.h"
#include "IO/IO.h"
#include "Pattern/Evaluator.h"
#include "PatternFit/PatternFit.h"
#include "PatternIO/PatternIO.h"
#include "Search/Search.h"
#include "GameIO/GameIO.h"
#include <execution>
#include <iostream>
#include <string>
#include <chrono>

using namespace std::chrono_literals;

IntensityTable train_requests;
IntensityTable test_requests;
IntensityTable accuracy_fit_requests;

std::size_t train_size = 100'000;
std::size_t test_size = 1'000;
std::size_t accuracy_fit_size = 500;

class TimerWithLog
{
	std::vector<std::chrono::high_resolution_clock::duration> log;
	std::chrono::high_resolution_clock::time_point start;
public:
	TimerWithLog() = default;

	void clear() { log.clear(); }
	void Start() { start = std::chrono::high_resolution_clock::now(); }
	void Stop() { log.push_back(std::chrono::high_resolution_clock::now() - start); }
	auto Log() const { return log; }
};

struct ModelQuality
{
	std::vector<double> stddev;
	double R_sq;
};

class IterationTable : public Table
{
	std::string iteration = "";
	int last_iteration = 99;
public:
	IterationTable(AAGLEM& model)
	{
		AddColumn("Iteration", 9, "{:>9}");
		for (HalfOpenInterval boundary : model.Boundaries())
			AddColumn(fmt::format("{:02}..", boundary.lower), 4, "{:4.2f}");
		AddColumn("R^2", 4, "{:4.2f}");
		AddColumn("Play", 6, "{:>6}");
		AddColumn("Lift", 6, "{:>6}");
		AddColumn("Acc", 6, "{:>6}");
	}

	void PrintRow(int iteration, ModelQuality quality, std::vector<std::chrono::high_resolution_clock::duration> durations)
	{
		int i = 0;
		Table::print_content(i++, iteration == last_iteration ? std::nullopt : std::optional(iteration));
		for (double sd : quality.stddev)
			Table::print_content(i++, sd);
		Table::print_content(i++, quality.R_sq);
		for (auto d : durations)
			Table::print_content(i++, iteration == last_iteration ? std::nullopt : std::optional(std::chrono::round<std::chrono::seconds>(d)));
		fmt::print("\n");

		last_iteration = iteration;
	}
};

class BenchmarkTable : public Table
{
public:
	BenchmarkTable()
	{
		AddColumn("empty_count", 11, "{:^11}");
		AddColumn("time", 7, "{:7}");
	}

	void PrintRow(int empty_count, auto duration) const
	{
		Table::PrintRow(empty_count, short_time_format(duration));
	}
};

Filebased<AAGLEM> LoadModel(std::string name)
{
	std::filesystem::path file = fmt::format(R"(G:\Reversi\{}.model)", name);
	if (std::filesystem::exists(file))
		return Filebased<AAGLEM>(file);
	return Filebased<AAGLEM>(file, AAGLEM{
		{
			//"- - - - - - - -"
			//"- - - - - - - -"
			//"- - - - - - - -"
			//"- - - - - - - -"
			//"- - - - - - - -"
			//"- - - - - - - -"
			//"- - - # # # # #"
			//"- - - # # # # #"_BitBoard,

			//"- - - - - - - -"
			//"- - - - - - - -"
			//"- - - - - - - -"
			//"- - - - - - - -"
			//"- - - - - - - #"
			//"- - - - - - # #"
			//"- - - - - # # #"
			//"- - - - # # # #"_BitBoard,

			//"- - - - - - - -"
			//"- - - - - - - -"
			//"- - - - - - - -"
			//"- - - - - - - -"
			//"- - - - - - - -"
			//"- - - - - - - -"
			//"- # - - - - # -"
			//"# # # # # # # #"_BitBoard,

			//"- - - - - - - -"
			//"- - - - - - - -"
			//"- - - - - - - -"
			//"- - - - - - - -"
			//"- - - - # # - -"
			//"- - - - # # # -"
			//"- - - - - # # #"
			//"- - - - - - # #"_BitBoard,

			//"# - - - - - - -"
			//"- # - - - - - -"
			//"- - # - - - - -"
			//"- - - # - - - -"
			//"- - - - # - - -"
			//"- - - - - # - -"
			//"- - - - - - # #"
			//"- - - - - - # #"_BitBoard,

			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - # # #"
			"- - - - - # # #"
			"- - - - - # # #"_BitBoard,

			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - #"
			"- - - - - - - #"
			"- - - - - - - #"
			"- - - - - - # #"
			"- - - # # # # #"_BitBoard,

			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - -"
			"- # - - - - # -"
			"# # # # # # # #"_BitBoard,

			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - -"
			"- - # # # # - -"
			"# - # # # # - #"_BitBoard,

			BitBoard::HorizontalLine(1), // L1
			BitBoard::HorizontalLine(2), // L2
			BitBoard::HorizontalLine(3), // L3
			BitBoard::CodiagonalLine(0), // D8
			BitBoard::CodiagonalLine(1), // D7
			BitBoard::CodiagonalLine(2), // D6
			BitBoard::CodiagonalLine(3), // D5
			BitBoard::CodiagonalLine(4), // D4
			}, { 0,5,10,15,20,25,30,35,40,45,50 } }
		//}, { 0,10,20,30,40,50 } }
	);
}

FilebasedDB<NoMovePuzzle> LoadDB(std::filesystem::path file)
{
	if (std::filesystem::exists(file))
		return FilebasedDB<NoMovePuzzle>(file);
	return FilebasedDB<NoMovePuzzle>(file, {}, false);
}

auto WhereEmptyCount(int value)
{
	return [value](const NoMovePuzzle& p) { return EmptyCount(p) == value; };
}

auto WhereEmptyCount(HalfOpenInterval value)
{
	return [value](const NoMovePuzzle& p) { return value.lower <= EmptyCount(p) and EmptyCount(p) < value.upper; };
}

auto to_max_intensity_score()
{
	return ranges::views::transform([](const NoMovePuzzle& p) { return p.ResultOf(p.MaxSolvedIntensity().value()); });
}

auto to_score(Intensity intensity)
{
	return ranges::views::transform([intensity](const NoMovePuzzle& p) { return p.ResultOf(intensity); });
}

auto to_pos_multi_depth_score()
{
	return ranges::views::transform(
		[](const NoMovePuzzle& p) {
			PosMultiDepthScore ret{ p.pos };
			for (const NoMovePuzzle::Task& t : p.tasks)
				if (t.IsCertain() and t.IsDone())
					ret.score_of_depth[t.Depth()] = t.Score();
			return ret;
		});
}

auto to_pos_best_score()
{
	return ranges::views::transform(
		[](const NoMovePuzzle& p) {
			return PosScore{ p.pos, p.ResultOf(p.MaxSolvedIntensity().value()) };
		});
}

std::vector<Position> CreateData(Algorithm& alg, Intensity depth1, Intensity depth2, int empty_count, std::size_t size)
{
	static std::mt19937_64 rnd_engine(std::random_device{}());
	static std::vector<Position> all = AllUnique(Position::Start(), 50);

	auto player1 = FixedDepthPlayer(alg, depth1);
	auto player2 = FixedDepthPlayer(alg, depth2);

	std::vector pos = generate_n_unique(std::execution::par, size, PosGen::Played(player1, player2, empty_count, all)) | ranges::to_vector;
	std::ranges::shuffle(pos, rnd_engine);
	return pos;
}

std::vector<Position> CreateData(Algorithm& alg, Intensity depth1, Intensity depth2, HalfOpenInterval empty_count, std::size_t size_each)
{
	std::vector<Position> ret;
	ret.reserve((empty_count.upper - empty_count.lower) * size_each);
	for (int e = empty_count.lower; e < empty_count.upper; e++)
	{
		auto data = CreateData(alg, depth1, depth2, e, size_each);
		ret.insert(ret.end(), data.begin(), data.end());
	}
	return ret;
}

std::vector<Game> CreateGames(Algorithm& alg, Intensity depth1, Intensity depth2, int size)
{
	static std::vector<Position> all = AllUnique(Position::Start(), 50);

	auto player1 = FixedDepthPlayer(alg, depth1);
	auto player2 = FixedDepthPlayer(alg, depth2);
	auto generator = GameGen::Played(player1, player2, all);

	std::vector<Game> games(size);
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < size; i++)
		games[i] = generator();
	return games;
}

auto to_puzzle(const IntensityTable& requests)
{
	return ranges::views::join
		| ranges::views::transform([&requests](NoMovePuzzle p)
			{
				p.insert(requests.Intensities(EmptyCount(p)));
				return p;
			});
}

//void aaa(Algorithm& alg, Intensity depth1, Intensity depth2, int train_size, int test_size, int accuracy_fit_size)
//{
//	DB<NoMovePuzzle> db;
//	std::vector<Game> games = CreateGames(alg, depth1, depth2, train_size + test_size + accuracy_fit_size);
//
//	auto train = games | ranges::views::take(train_size) | to_puzzle(train_requests);
//	auto test = games | ranges::views::drop(train_size) | ranges::views::take(test_size) | to_puzzle(test_requests);
//	auto accuracy_fit = games | ranges::views::drop(train_size) | ranges::views::drop(test_size) | to_puzzle(accuracy_fit_requests);
//
//	db.Add({ "train" }, train);
//	db.Add({ "test" }, test);
//	db.Add({ "accuracy_fit" }, accuracy_fit);
//}
//
//std::vector<Position> CreateData(Algorithm& alg, Intensity depth1, Intensity depth2, int size)
//{
//	static std::vector<Position> all = AllUnique(Position::Start(), 50);
//
//	auto player1 = FixedDepthPlayer(alg, depth1);
//	auto player2 = FixedDepthPlayer(alg, depth2);
//	auto generator = GameGen::Played(player1, player2, all);
//
//	std::vector<Game> games(size);
//	#pragma omp parallel for schedule(static)
//	for (int i = 0; i < size; i++)
//		games[i] = generator();
//
//	std::vector<Position> pos;
//	for (const Game& game : games)
//		for (const Position& p : game)
//			pos.push_back(p);
//	return pos;
//}
//
struct DataSet
{
	std::vector<NoMovePuzzle> train, test, accuracy_fit;

	void Add(const DataSet& o)
	{
		train.insert(train.end(), o.train.begin(), o.train.end());
		test.insert(test.end(), o.test.begin(), o.test.end());
		accuracy_fit.insert(accuracy_fit.end(), o.accuracy_fit.begin(), o.accuracy_fit.end());
	}
};

DataSet CreateData(Intensity depth1, Intensity depth2, Algorithm& alg)
{
	//HalfOpenInterval empty_count{ 0, 50 };
	std::vector<Game> games = CreateGames(alg, depth1, depth2, train_size + test_size + accuracy_fit_size);

	DataSet ret;
	ret.train.reserve(train_size);
	ret.test.reserve(test_size);
	ret.accuracy_fit.reserve(accuracy_fit_size);
	for (int i = 0; i < train_size; i++)
		for (NoMovePuzzle p : games[i])
		{
			p.insert(train_requests.Intensities(EmptyCount(p)));
			ret.train.push_back(p);
		}
	for (int i = 0; i < test_size; i++)
		for (NoMovePuzzle p : games[train_size + i])
		{
			p.insert(test_requests.Intensities(EmptyCount(p)));
			ret.test.push_back(p);
		}
	for (int i = 0; i < accuracy_fit_size; i++)
		for (NoMovePuzzle p : games[train_size + test_size + i])
		{
			p.insert(accuracy_fit_requests.Intensities(EmptyCount(p)));
			ret.accuracy_fit.push_back(p);
		}
	return ret;
}

DataSet CreateData(std::vector<std::pair<Intensity, Intensity>> depths, Algorithm& alg)
{
	DataSet ret;
	for (std::pair<Intensity, Intensity> depth : depths)
		ret.Add(CreateData(depth.first, depth.second, alg));
	return ret;
}

void AddData(DB<NoMovePuzzle>& db, const DataSet& data, const std::vector<std::string>& metas)
{
	using namespace std::string_literals;
	db.Add(ranges::views::concat(metas, ranges::single_view("train"s)), data.train);
	db.Add(ranges::views::concat(metas, ranges::single_view("test"s)), data.test);
	db.Add(ranges::views::concat(metas, ranges::single_view("accuracy_fit"s)), data.accuracy_fit);
}


//void DuplicateAllData(DB<NoMovePuzzle>& db, std::string old_meta, std::string new_meta)
//{
//	db.Add({ new_meta, "train" }, db.WhereAllOf({ old_meta, "train" }));
//	db.Add({ new_meta, "test" }, db.WhereAllOf({ old_meta, "test" }));
//	db.Add({ new_meta, "accuracy_fit" }, db.WhereAllOf({ old_meta, "accuracy_fit" }));
//}

std::chrono::milliseconds Solve(auto&& data, Algorithm& alg, std::function<void(void)> write_back)
{
	auto ex = CreateExecutor(std::execution::par, data, [&](NoMovePuzzle& p) { p.Solve(alg); });

	Metronome auto_saver(
		300s,
		[&] {
			ex->LockData();
			write_back;
			ex->UnlockData();
		}
	);

	auto_saver.Start();
	auto start = std::chrono::high_resolution_clock::now();
	ex->Process();
	auto stop = std::chrono::high_resolution_clock::now();
	auto_saver.Stop();
	auto_saver.Force();
	return round<std::chrono::milliseconds>(stop - start);
}

std::chrono::milliseconds Solve(auto&& data, Algorithm& alg)
{
	auto start = std::chrono::high_resolution_clock::now();
	Process(std::execution::par, data, [&](NoMovePuzzle& p) { p.Solve(alg); });
	auto stop = std::chrono::high_resolution_clock::now();
	return round<std::chrono::milliseconds>(stop - start);
}

void EvaluateInexacts(auto&& data, Algorithm& alg)
{
	for (NoMovePuzzle& p : data)
		p.ClearInexacts();
	Solve(data, alg);
}

std::pair<AAGLEM, DataSet> Iteration(auto&& old_data, std::vector<std::pair<Intensity, Intensity>> depths, const AAGLEM& old_model, TimerWithLog& timer)
{
	AAGLEM new_model = old_model;
	HT tt_1{ 10'000'000 };
	HT tt_2{ 10'000'000 };
	auto old_alg = PVS{ tt_1, old_model };
	auto new_alg = PVS{ tt_2, new_model };

	timer.Start();
	DataSet new_data = CreateData(depths, old_alg);
	timer.Stop();

	timer.Start();
	for (std::size_t block = 0; block < new_model.Blocks(); block++)
	{
		auto in_block = ranges::views::filter(WhereEmptyCount(new_model.Boundaries(block)));

		Solve(new_data.train | in_block, new_alg);

		// Fit evaluation model
		auto new_weights = FittedWeights(new_model.Pattern(), ranges::views::concat(old_data, new_data.train) | in_block | to_pos_best_score());
		new_model.SetWeights(block, new_weights);
	}
	timer.Stop();

	timer.Start();
	EvaluateInexacts(new_data.accuracy_fit, new_alg);
	new_model.AccuracyModel() = FittedAM(new_data.accuracy_fit | to_pos_multi_depth_score());
	timer.Stop();

	return std::make_pair(new_model, new_data);
}

ModelQuality EvaluateModel(const AAGLEM& model, auto&& data)
{
	HT tt{ 10'000'000 };
	auto alg = PVS{ tt, model };

	for (NoMovePuzzle& datum : data)
		datum.ClearResult(0);

	Solve(data, alg);

	ModelQuality ret;
	for (std::size_t block = 0; block < model.Blocks(); block++)
	{
		auto data_in_block = data | ranges::views::filter(WhereEmptyCount(model.Boundaries(block))) | ranges::views::filter([](const NoMovePuzzle& p) { return std::abs(p.ResultOf(0)) < 27; });
		auto data_error = ranges::views::transform(
			data_in_block | to_max_intensity_score(),
			data_in_block | to_score(Intensity(0)),
			std::minus()
		);
		ret.stddev.push_back(StandardDeviation(data_error));
	}
	ret.R_sq = 0.0; // TODO!
	return ret;
}

void Benchmark(const AAGLEM& model, auto&& data, HalfOpenInterval empty_count)
{
	HT tt{ 100'000'000 };
	auto alg = IDAB<PVS>{ tt, model };
	BenchmarkTable table;
	table.PrintHeader();
	for (int e = min; e <= max; e++)
	{
		auto duration = Solve_exacts(data.Where(WhereEmptyCount(e)), alg);
		table.PrintRow(e, duration);
	}
	table.PrintSeparator();
}


void Test(const AAGLEM& model, const range<PosScore> auto& pos_scores)
{
	HT tt{ 1'000'000 };
	auto alg = IDAB<PVS>{ tt, model };

	SolverTable table(/*test*/ true);
	table.PrintHeader();

	std::vector<int> diff;
	std::chrono::nanoseconds duration{ 0 };
	uint64 nodes = 0;
	for (auto&& [i, ps] : ranges::views::enumerate(pos_scores))
	{
		auto start = std::chrono::high_resolution_clock::now();
		int eval = alg.Eval(ps.pos);
		auto stop = std::chrono::high_resolution_clock::now();
		auto dur = stop - start;

		diff.push_back(eval - ps.score);
		duration += dur;
		nodes += alg.Nodes();

		table.PrintRow(i, EmptyCount(ps), ps.score, eval, dur.count(), alg.Nodes());
	}
	table.PrintSeparator();
	table.PrintSummary(StandardDeviation(diff), duration.count(), nodes);
}

int main()
{
	train_requests = IntensityTable::ExactTill(14, /*then*/5);
	test_requests = IntensityTable::ExactTill(14, /*then*/1); test_requests.insert(0);
	accuracy_fit_requests = IntensityTable::AllDepthTill(14, /*then*/ std::vector{ 0,1,2,3,4,5,6,7,8 });


	AAGLEM model_0{
		{
			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - # # #"
			"- - - - - # # #"
			"- - - - - # # #"_BitBoard,

			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - #"
			"- - - - - - - #"
			"- - - - - - - #"
			"- - - - - - # #"
			"- - - # # # # #"_BitBoard,

			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - -"
			"- # - - - - # -"
			"# # # # # # # #"_BitBoard,

			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - -"
			"- - - - - - - -"
			"- - # # # # - -"
			"# - # # # # - #"_BitBoard,

			BitBoard::HorizontalLine(1), // L1
			BitBoard::HorizontalLine(2), // L2
			BitBoard::HorizontalLine(3), // L3
			BitBoard::CodiagonalLine(0), // D8
			BitBoard::CodiagonalLine(1), // D7
			BitBoard::CodiagonalLine(2), // D6
			BitBoard::CodiagonalLine(3), // D5
			BitBoard::CodiagonalLine(4), // D4
		}, { 0,5,10,15,20,25,30,35,40,45,50 }
	};

	DB<NoMovePuzzle> db;// = Deserialize<DB<NoMovePuzzle>>(R"(G:\Reversi\alpha_zero2.db)");
	IterationTable table(model_0);
	TimerWithLog timer;
	AAGLEM model_1;

	//{
	//	Filebased<AAGLEM> model(R"(G:\Reversi\d5_d5_model_40.mdl)");
	//	HT tt{ 10'000'000 };
	//	auto alg = PVS{ tt, model };
	//	auto data = db.WhereAllOf({ "d5_d5", "it_40", "accuracy_fit" });
	//	EvaluateInexacts(data, alg);
	//	model.AccuracyModel() = FittedAM(data | to_pos_multi_depth_score());
	//	model.WriteBack();
	//	return 0;
	//}

	table.PrintHeader();

	{
		std::pair<AAGLEM, DataSet> it_1 = Iteration(std::vector<NoMovePuzzle>{}, { {0,0} }, model_0, timer);
		model_1 = it_1.first;
		Serialize(model_1, R"(G:\Reversi\d0_d0_model_1.mdl)");
		AddData(db, it_1.second, { "d0_d0", "it_1" });
	}

	auto quality = EvaluateModel(model_1, db.Where("test"));
	table.PrintRow(1, quality, timer.Log());

	//for (int d = 0; d <= 4; d++)
	//	for (int D = d; D <= 4; D++)
		{
			int d = 5;
			int D = 5;
			std::string dD = fmt::format("d{}_d{}", d, D);

			std::vector<AAGLEM> models;
			models.push_back(model_1);

			for (int i = 2; i <= 100; i++)
			{
				std::string it_name = fmt::format("it_{}", i);
				timer.clear();
				{
					auto old_data = ranges::views::concat(
						db.WhereAllOf({ "d0_d0", "it_1", "train" }),
						db.WhereAllOf({ dD, "train" }));
					auto [model, new_data] = Iteration(old_data, { {d,D}/*, {D,d}*/ }, models.back(), timer);
					Serialize(model, fmt::format(R"(G:\Reversi\d{}_d{}_model_{}.mdl)", d, D, i));
					AddData(db, new_data, { dD, it_name });
					models.push_back(model);
				}
				for (const AAGLEM& model : models)
				{
					auto quality = EvaluateModel(model, db.WhereAllOf({ dD, "test" }));
					table.PrintRow(i, quality, timer.Log());
				}
				table.PrintSeparator();
			}
			Serialize(db, R"(G:\Reversi\alpha_zero2.db)");
			table.PrintSeparator();
		}

	//table.PrintHeader();

	//quality = EvaluateModel(Deserialize<AAGLEM>(R"(G:\Reversi\d0_d0_model_1.mdl)"), db.Where("test"));
	//table.PrintRow(1, quality, {});

	//for (int d = 0; d <= 4; d++)
	//	for (int D = d; D <= 4; D++)
	//		for (int i = 2; i <= 4; i++)
	//		{
	//			auto model = Deserialize<AAGLEM>(fmt::format(R"(G:\Reversi\d{}_d{}_model_{}.mdl)", d, D, i));
	//			auto quality = EvaluateModel(model, db.Where("test"));
	//			table.PrintRow(d * 10 + D, quality, {});
	//		}
	return 0;
}