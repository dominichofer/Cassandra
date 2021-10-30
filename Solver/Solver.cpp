#include "Core/Core.h"
#include "Search/Search.h"
#include "IO/IO.h"
#include "IO/Integers.h"
#include "Math/Algorithm.h"
#include "Math/Statistics.h"
#include "Pattern/Evaluator.h"
#include "PatternFit/PatternFit.h"

#include <array>
#include <chrono>
#include <numeric>
#include <vector>
#include <iostream>
#include <iomanip>
#include <set>
#include <format>
#include <omp.h>
#include <experimental/generator>

class OutputFormater
{
	bool test;
	std::string Fmt(int diff) const
	{
		std::string s;
		s += "{:>9}|"; // index
		s += "{:<6}|"; // depth
		s += " {:+03} |"; // eval
		if (test)
		{
			s += " {:+03} |"; // score
			if (diff)
				s += " {:+03} |"; // diff
			else
				s += "   {:1}  |"; // diff
		}
		else
		{
			s += "{:0}"; // score
			s += "{:0}"; // diff
		}
		s += "{:>16.3}|"; // time
		s += "{:>18}|"; // nodes
		s += "{:>14.0}"; // N/s
		return s + '\n';
	}
public:
	OutputFormater(bool test) noexcept : test(test) {}

	std::string Header() const
	{
		if (test)
			return "  # | depth  | eval|score| diff|       time (s) |        nodes (N) |             N/s\n" + Footer();
		return "  # | depth  | eval|       time (s) |        nodes (N) |             N/s\n" + Footer();
	}

	std::string Footer() const
	{
		if (test)
			return "----+--------+-----+-----+-----+----------------+------------------+-----------------\n";
		return "----+--------+-----+----------------+------------------+-----------------\n";
	}

	std::string Line(std::size_t index, Intensity intensity, int eval, int score, std::chrono::duration<double> time, std::size_t nodes)
	{
		int diff = eval - score;
		auto nps = nodes / time.count();

		std::string s = std::format("{:>3} | {:<6} | {:+03} |", index, to_string(intensity), eval);
		if (test)
		{
			s += std::format(" {:+03} |", score);
			if (diff)
				s += std::format(" {:+03} |", diff);
			else
				s += "     |";
		}
		return s + std::format(std::locale(""), "{:15.3f} |{:>17L} |{:>14.0Lf}\n", time.count(), nodes, nodes / time.count());
	}
	std::string Line(std::size_t index, const Request& request, int score, const Result& result)
	{
		return Line(index, request.intensity, score, result.score, result.duration, result.nodes);
	}
	std::string Line(std::size_t index, Intensity intensity, int eval, std::chrono::duration<double> time, std::size_t nodes)
	{
		assert(not test);
		return Line(index, intensity, eval, 0 /*ignored*/, time, nodes);
	}
	std::string Line(std::size_t index, const Request& request, const Result& result)
	{
		return Line(index, request.intensity, result.score, result.duration, result.nodes);
	}

	std::string Summary(int diff, std::chrono::duration<double> time, std::size_t nodes)
	{
		std::string s = test ? "                         " : "             ";
		if (test and diff)
			s += std::format("| {:+03} |", diff);
		else
			s += "|     |";
		return s + std::format(std::locale(""), "{:15.3f} |{:>17L} |{:>14.0Lf}\n", time.count(), nodes, nodes / time.count());
	}
};

using namespace std::chrono_literals;

void Test(PuzzleRange auto puzzles)
{
	AAGLEM pattern_eval = DefaultPatternEval();
	HashTablePVS tt{ 1'000'000 };

	std::vector<int> diff;
	OutputFormater formater(true);
	std::cout << formater.Header();
	Process(std::execution::seq, puzzles,
		[&](Puzzle& p, std::size_t index) {
			auto exact = Request::ExactScore(p.pos);
			int old_score = p.ResultOf(exact).score;
			p.ClearResult(exact);
			p.Solve(IDAB{ tt, pattern_eval });
			diff.push_back(old_score - p.ResultOf(exact).score);
			std::cout << formater.Line(index, exact, old_score, p.ResultOf(exact));
		});
	std::cout << formater.Footer();
	std::cout << formater.Summary(StandardDeviation(diff), Duration(puzzles), Nodes(puzzles));
}

auto CreateNewData(int d1, int d2)
{
	std::string eval_fit_name = std::format(R"(G:\Reversi\play{}{}_eval_fit.puz)", d1, d2);
	std::string accuracy_fit_name = std::format(R"(G:\Reversi\play{}{}_accuracy_fit.puz)", d1, d2);
	std::string move_sort_name = std::format(R"(G:\Reversi\play{}{}_move_sort.puz)", d1, d2);
	std::string benchmark_name = std::format(R"(G:\Reversi\play{}{}_benchmark.puz)", d1, d2);

	if (not std::filesystem::exists(eval_fit_name))
	{
		HashTablePVS tt1{ 100'000'000 };
		HashTablePVS tt2{ 100'000'000 };
		AAGLEM pattern_eval = DefaultPatternEval();
		std::mt19937_64 rnd_engine(std::random_device{}());

		auto start = std::chrono::high_resolution_clock::now();
		std::vector<Position> all = AllUnique(Position::Start(), 50);

		std::unique_ptr<Player> player1;
		if (d1 == 0)
			player1 = std::make_unique<RandomPlayer>();
		else
			player1 = std::make_unique<FixedDepthPlayer>(tt1, pattern_eval, d1);

		std::unique_ptr<Player> player2;
		if (d2 == 0)
			player2 = std::make_unique<RandomPlayer>();
		else
			player2 = std::make_unique<FixedDepthPlayer>(tt2, pattern_eval, d2);

		std::vector<Puzzle> eval_fit, accuracy_fit, move_sort, benchmark;
		for (int e = 0; e <= 50; e++)
		{
			auto set = generate_n_unique(102'500, PosGen::Played(*player1, *player2, e, all));
			std::vector<Position> pos(set.begin(), set.end());
			std::shuffle(pos.begin(), pos.end(), rnd_engine);
			auto it1 = pos.begin();
			auto it2 = it1 + 100'000;
			auto it3 = it2 + 500;
			auto it4 = it3 + 1'000;
			auto it5 = it4 + 1'000;

			eval_fit.insert(eval_fit.end(), it1, it2);
			accuracy_fit.insert(accuracy_fit.end(), it2, it3);
			move_sort.insert(move_sort.end(), it3, it4);
			benchmark.insert(benchmark.end(), it4, it5);
			std::cout << e << ' ';
		}

		Save(eval_fit_name, eval_fit);
		Save(accuracy_fit_name, accuracy_fit);
		Save(move_sort_name, move_sort);
		Save(benchmark_name, benchmark);
		auto stop = std::chrono::high_resolution_clock::now();
		std::cout << "\nCreateNewData(" << d1 << ", " << d2 << ") took " << std::chrono::duration_cast<std::chrono::seconds>(stop - start) << '\n';
	}
	return std::make_tuple(eval_fit_name, accuracy_fit_name, move_sort_name, benchmark_name);
}

void SolveEvalFit(DataBase<Puzzle>& data, int exact_till, int depth)
{
	const auto block_size = DefaultPatternEval().block_size;

	for (int block = 0; block < 50 / block_size; block++)
	{
		AAGLEM pattern_eval = DefaultPatternEval();
		HashTablePVS tt{ 100'000'000 };
		auto start = std::chrono::high_resolution_clock::now();
		auto IsInBlock = [block, block_size](const Puzzle& p) {
			int E = p.pos.EmptyCount();
			int lowest_E = 1 + block_size * block;
			return (lowest_E <= E) and (E < lowest_E + block_size);
		};

		auto ex = CreateExecutor(std::execution::par,
			data | std::views::filter(IsInBlock),
			[&](Puzzle& p, std::size_t index) {
				if (p.pos.EmptyCount() <= exact_till)
					p.insert(Request::ExactScore(p.pos));
				else
					p.insert(Request(depth));
				p.Solve(IDAB{ tt, pattern_eval });
			});

		Metronome auto_saver(300s, [&] {
			std::cout << "Saving...";
			ex->LockData();
			data.WriteBack();
			ex->UnlockData();
			std::cout << "done! Processed " << ex->Processed() << " of " << ex->size() << '\n';
			});

		auto_saver.Start();
		ex->Process();
		auto_saver.Stop();
		auto_saver.Force();
		FitWeights(data, block, true);
		auto stop = std::chrono::high_resolution_clock::now();
		std::cout << "SolveEvalFit(" << exact_till << ", " << depth << ") of block " << block << " took " << std::chrono::duration_cast<std::chrono::seconds>(stop - start) << '\n';
	}
}

class CartesianPoint;
double norm(const CartesianPoint&);
CartesianPoint operator/(CartesianPoint, double);

class CartesianPoint
{
	std::vector<double> x;
public:
	CartesianPoint(int dims) noexcept : x(dims, 0.0) {}

	int size() const noexcept { return x.size(); }
	      double& operator[](int i)       { return x[i]; }
	const double& operator[](int i) const { return x[i]; }
	decltype(auto) begin() noexcept { return x.begin(); }
	decltype(auto) begin() const noexcept  { return x.begin(); }
	decltype(auto) end() noexcept { return x.end(); }
	decltype(auto) end() const noexcept { return x.end(); }
	decltype(auto) front() const noexcept { return x.front(); }
	decltype(auto) back() const noexcept { return x.back(); }
	CartesianPoint normalized() const noexcept
	{
		double n = norm(*this);
		if (n)
			return *this / n;
		return *this;
	}

	CartesianPoint& operator+=(const CartesianPoint& o)
	{
		assert(size() == o.size());
		#pragma omp parallel for schedule(static)
		for (int64_t i = 0; i < static_cast<int64_t>(size()); i++)
			x[i] += o[i];
		return *this;
	}

	CartesianPoint& operator-=(const CartesianPoint& o)
	{
		assert(size() == o.size());
		#pragma omp parallel for schedule(static)
		for (int64_t i = 0; i < static_cast<int64_t>(size()); i++)
			x[i] -= o[i];
		return *this;
	}

	CartesianPoint& operator*=(const CartesianPoint& o)
	{
		assert(size() == o.size());
		#pragma omp parallel for schedule(static)
		for (int64_t i = 0; i < static_cast<int64_t>(size()); i++)
			x[i] *= o[i];
		return *this;
	}

	CartesianPoint& operator*=(double m)
	{
		#pragma omp parallel for schedule(static)
		for (int64_t i = 0; i < static_cast<int64_t>(size()); i++)
			x[i] *= m;
		return *this;
	}

	CartesianPoint& operator/=(double m)
	{
		#pragma omp parallel for schedule(static)
		for (int64_t i = 0; i < static_cast<int64_t>(size()); i++)
			x[i] /= m;
		return *this;
	}
};

CartesianPoint operator+(CartesianPoint l, const CartesianPoint& r) { return l += r; }
CartesianPoint operator+(const CartesianPoint& l, CartesianPoint&& r) { return r += l; }

CartesianPoint operator-(CartesianPoint l, const CartesianPoint& r) { return l -= r; }
CartesianPoint operator-(const CartesianPoint& l, CartesianPoint&& r)
{
	assert(l.size() == r.size());
	const int64_t size = static_cast<int64_t>(r.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		r[i] = l[i] - r[i];
	return r;
}

CartesianPoint operator*(CartesianPoint l, const CartesianPoint& r) { return l *= r; }
CartesianPoint operator*(const CartesianPoint& l, CartesianPoint&& r) { return r *= l; }
CartesianPoint operator*(CartesianPoint vec, double mul) { return vec *= mul; }
CartesianPoint operator*(double mul, CartesianPoint vec) { return vec *= mul; }

CartesianPoint operator/(CartesianPoint vec, double mul) { return vec /= mul; }

double dot(const CartesianPoint& l, const CartesianPoint& r)
{
	assert(l.size() == r.size());

	const int64_t size = static_cast<int64_t>(l.size());
	double sum = 0.0;
	#pragma omp parallel for reduction(+ : sum)
	for (int64_t i = 0; i < size; i++)
		sum += l[i] * r[i];
	return sum;
}

double L0(const CartesianPoint& x)
{
	const int64_t size = static_cast<int64_t>(x.size());
	double sum = 0.0;
	#pragma omp parallel for reduction(+ : sum)
	for (int64_t i = 0; i < size; i++)
		sum += (x[i] < 0.001 ? 0 : 1);
	return sum;
}

double L1(const CartesianPoint& x)
{
	const int64_t size = static_cast<int64_t>(x.size());
	double sum = 0.0;
	#pragma omp parallel for reduction(+ : sum)
	for (int64_t i = 0; i < size; i++)
		sum += std::abs(x[i]);
	return sum;
}

double L2(const CartesianPoint& x)
{
	return std::sqrt(dot(x, x));
}

double norm(const CartesianPoint& x)
{
	return L2(x);
}

std::string to_string(const CartesianPoint& vec)
{
	using std::to_string;
	std::string s;
	for (int i = 0; i < vec.size() - 1; i++)
		s += to_string(vec[i]) + ", ";
	return s + to_string(vec.back());
}

CartesianPoint RandomPointInCube(int dims, double min, double max, std::mt19937_64& rnd_engine)
{
	std::uniform_real_distribution<double> dist(min, max);
	CartesianPoint p(dims);
	for (int d = 0; d < dims; d++)
		p[d] = dist(rnd_engine);
	return p;
}
CartesianPoint RandomPointOnCubeSurface(int dims, double min, double max, std::mt19937_64& rnd_engine)
{
	CartesianPoint p = RandomPointInCube(dims, min, max, rnd_engine);

	int special_dim = std::uniform_int_distribution<int>(0, dims - 1)(rnd_engine);
	bool min_or_max = (std::uniform_int_distribution<int>(0, 1)(rnd_engine) == 0);
	p[special_dim] = (min_or_max ? min : max);
	return p;
}

double Value(const Puzzle& puzzle, const CartesianPoint& w, std::function<double(const Position&, const CartesianPoint&)> metric)
{
	std::optional<Intensity> opt_max = puzzle.MaxSolvedIntensityOfAllMoves();
	if (not opt_max.has_value())
		return 0;
	Intensity max_intensity = opt_max.value();

	struct MV {
		Field move;
		double value;
	};
	auto pm = PossibleMoves(puzzle.pos);
	std::vector<MV> sorted_moves;
	sorted_moves.reserve(pm.size());
	for (Field move : pm)
		sorted_moves.emplace_back(move, metric(Play(puzzle.pos, move), w));
	std::sort(sorted_moves.begin(), sorted_moves.end(), [](const MV& l, const MV& r) { return l.value < r.value; });

	std::set<Field> best_moves = puzzle.BestMoves();

	double total_nodes = 0;
	double searched_nodes = 0;
	bool found_best_move = false;
	for (const auto& mv : sorted_moves)
	{
		uint64_t nodes = puzzle.ResultOf(Request{ mv.move, max_intensity - 1 }).nodes;
		total_nodes += nodes;
		if (not found_best_move)
			searched_nodes += nodes;
		if (best_moves.contains(mv.move))
			found_best_move = true;
	}
	return searched_nodes / total_nodes;
}

double Value(PuzzleRange auto&& puzzles, const CartesianPoint& w, std::function<double(const Position&, const CartesianPoint&)> metric)
{
	return std::transform_reduce(
		std::execution::par,
		puzzles.begin(), puzzles.end(),
		0.0, std::plus(), [&](const Puzzle& p) { return Value(p, w, metric); });
}

double OldMetric(const Position& pos, const CartesianPoint&)
{
	double score = 0; // FieldValue[static_cast<uint8_t>(move)];
	score += DoubleCornerPopcount(EightNeighboursAndSelf(pos.Opponent()) & pos.Empties()) * (1.0 / 1024.0);
	score += popcount(EightNeighboursAndSelf(pos.Empties()) & pos.Opponent()) * (1.0 / 512.0);
	score += DoubleCornerPopcount(PossibleMoves(pos));
	score += DoubleCornerPopcount(StableCornersOpponent(pos)) * -(1.0 / 16.0); // w_edge_stability
	return score;
}

double Metric(const Position& pos, const CartesianPoint& weights)
{
	auto w = weights.begin();
	double sum = 0;

	//sum += popcount(pos.Player() & BitBoard::Corners()) * *w++;
	//sum += popcount(pos.Opponent() & BitBoard::Corners()) * *w++;
	//sum += popcount(pos.Empties() & BitBoard::Corners()) * *w++;

	//sum += popcount(EightNeighboursAndSelf(pos.Player())) * *w++;
	//sum += popcount(EightNeighboursAndSelf(pos.Opponent())) * *w++;
	//sum += popcount(EightNeighboursAndSelf(pos.Empties())) * *w++;

	//+ popcount(EightNeighboursAndSelf(pos.Player()) & pos.Player()) // == EightNeighboursAndSelf(pos.Player())
	sum += popcount(EightNeighboursAndSelf(pos.Opponent()) & pos.Player()) * *w++;
	sum += popcount(EightNeighboursAndSelf(pos.Empties()) & pos.Player()) * *w++;

	sum += popcount(EightNeighboursAndSelf(pos.Player()) & pos.Opponent()) * *w++;
	//+ popcount(EightNeighboursAndSelf(pos.Opponent()) & pos.Opponent()) // == EightNeighboursAndSelf(pos.Opponent())
	sum += popcount(EightNeighboursAndSelf(pos.Empties()) & pos.Opponent()) * *w++;

	sum += popcount(EightNeighboursAndSelf(pos.Player()) & pos.Empties()) * *w++;
	sum += popcount(EightNeighboursAndSelf(pos.Opponent()) & pos.Empties()) * *w++;
	//+ popcount(EightNeighboursAndSelf(pos.Empties()) & pos.Empties()) // == EightNeighboursAndSelf(pos.Empties())

	//sum += popcount(StableCornersOpponent(pos)) * *w++;
	sum += popcount(StableEdges(pos) & pos.Opponent()) * *w++;

	auto pm = PossibleMoves(pos);
	sum += pm.size();
	sum += (pm & BitBoard::Corners()).size();
	return sum;
}

void ImproveOn(PuzzleRange auto&& puzzles, double best_value, CartesianPoint best_w, std::mt19937_64& rnd_engine)
{
	for (int i = 0; i < 500; i++)
	{
		CartesianPoint novum = best_w + RandomPointInCube(best_w.size(), -0.001, +0.001, rnd_engine);
		//novum /= norm(novum);
		auto novum_value = Value(puzzles, novum, &Metric);
		//std::cout << novum_value << ": " << to_string(novum) << '\n';
		if (novum_value <= best_value)
		{
			best_value = novum_value;
			best_w = novum;
			std::cout << best_value << ": " << to_string(best_w) << '\n';
		}
	}
	std::cout << best_value << ": " << to_string(best_w) << '\n';
}

void FindBestMoveSortWeights(PuzzleRange auto&& puzzles)
{
	std::mt19937_64 rnd_engine(std::random_device{}());

	CartesianPoint best_w = RandomPointInCube(7, -2, 2, rnd_engine);
	double best_value = Value(puzzles, best_w, &OldMetric);
	std::cout << best_value << '\n';

	for (int i = 0; i < 1'000'000; i++)
	{
		auto w = RandomPointInCube(best_w.size(), -2, 2, rnd_engine);
		//w /= norm(w);
		double value = Value(puzzles, w, &Metric);
		if (value < best_value)
		{
			best_value = value;
			best_w = w;
			//std::cout << best_value << ": " << to_string(best_w) << '\n';
			ImproveOn(puzzles, value, w, rnd_engine);
		}
	}
}

void SolveSomeAccuracyFit()
{
	std::cout.imbue(std::locale(""));

	HashTablePVS tt{ 1'000'000'000 };
	AAGLEM evaluator = DefaultPatternEval();
	DataBase<Puzzle> puzzles = LoadAccuracyFit();

	for (int e = 0; e <= 50; e++)
	{
		auto ex = CreateExecutor(std::execution::par,
			puzzles | std::views::filter([e](const Puzzle& p) { return p.pos.EmptyCount() == e; }),
			[&](Puzzle& p, std::size_t index) {
				p.insert(Request::ExactScore(p.pos));
				bool had_work = p.Solve(IDAB{ tt, evaluator });
				if (had_work)
					std::cout << index << ": " << to_string(p) << '\n';
			});

		Metronome auto_saver(60s, [&] {
			std::cout << "Saving...";
			ex->LockData();
			puzzles.WriteBack();
			ex->UnlockData();
			std::cout << "done! " << ex->Processed() << " of " << ex->size() << '\n';
			});

		auto_saver.Start();
		ex->Process();
		auto_saver.Stop();
		auto_saver.Force();
	}
}

int main(int argc, char* argv[])
{
	Test(FForum_1); std::cout << '\n';
	Test(FForum_2); std::cout << '\n';
	Test(FForum_3); std::cout << '\n';
	Test(FForum_4); std::cout << '\n';
	return 0;
	//DataBase<Puzzle> move_sort = LoadMoveSort();
	//FindBestMoveSortWeights(move_sort | std::views::filter([](const Puzzle& p) { return p.pos.EmptyCount() >= 10; }));
	//return 0;


	//FindBestMoveSortWeights(move_sort);
	//return 0;

	//HashTablePVS tt{ 100'000'000 };
	//AAGLEM evaluator = DefaultPatternEval();
	//DataBase<Puzzle> move_sort = LoadMoveSort();

	//for (int e = 0; e <= 50; e++)
	//{
	//	auto ex = CreateExecutor(std::execution::par,
	//		move_sort | std::views::filter([e](const Puzzle& p) { return p.pos.EmptyCount() == e; }),
	//		[&](Puzzle& p, std::size_t index) {
	//			for (Request r : Request::AllMoves(p.pos))
	//				p.insert(r);
	//			bool had_work = p.Solve(IDAB{ tt, evaluator });
	//			if (had_work)
	//				std::cout << index << ": " << to_string(p) << '\n';
	//		});

	//	Metronome auto_saver(60s, [&] {
	//		std::cout << "Saving...";
	//		ex->LockData();
	//		move_sort.WriteBack();
	//		ex->UnlockData();
	//		std::cout << "done! " << ex->Processed() << " of " << ex->size() << '\n';
	//		});

	//	auto_saver.Start();
	//	ex->Process();
	//	auto_saver.Stop();
	//	auto_saver.Force();
	//}
	//return 0;
	//Test(FForum_1); std::cout << '\n';
	//Test(FForum_2); std::cout << '\n';

	/*std::cout.imbue(std::locale(""));

	int i = 0;
	std::vector<std::pair<int, int>> ds = { {0,5}, {5,0}, {1,5}, {5,1}, {2,5}, {5,2}, {3,5}, {5,3}, {4,5}, {5,4} };
	for (auto d : ds)
	{
		auto [eval_fit_name, accuracy_fit_name, move_sort_name, benchmark_name] = CreateNewData(d.first, d.second);

		eval_fit.Add(eval_fit_name);
		accuracy_fit.Add(accuracy_fit_name);

		if (i++ % 3 == 0)
			for (Puzzle& p : eval_fit)
				p.ClearResult(Request(10));
		SolveEvalFit(eval_fit, 20, 10);

		HashTablePVS tt{ 100'000'000 };
		AAGLEM evaluator = DefaultPatternEval();
		auto start = std::chrono::high_resolution_clock::now();
		FitPattern(eval_fit, accuracy_fit, tt, evaluator);
		accuracy_fit.WriteBack();
		auto stop = std::chrono::high_resolution_clock::now();
		std::cout << "FitPattern() took " << std::chrono::duration_cast<std::chrono::seconds>(stop - start) << '\n';

		Test(FForum_1); std::cout << '\n';
		Test(FForum_2); std::cout << '\n';
	}
	return 0;*/

	//FitAccuracyModel();
	//return 0;

	//while (not project.IsSolved())
	//{
	//	#pragma omp parallel for
	//	for (int i = 0; i < 1000; i++)
	//		project.SolveNext(algorithm);

	//	for (int i = 0; i < pos_score.size(); i++)
	//		if (project[i].IsSolved())
	//			pos_score[i].score = project[i].Score();
	//	Save(file_name, pos_score);
	//	std::cout << "Saved!" << std::endl;
	//}

	// Postprocessing
	//std::vector<int> score_diff;
	//for (const Puzzle& puzzle : project.Puzzles())
	//	score_diff.push_back(puzzle.Result(0).Score() - puzzle.Result().Score());
	//const auto correct = std::count_if(score_diff.begin(), score_diff.end(), [](int i) { return i == 0; });

	//std::cout << "---+------+-----+-----+-----+----------------+----------------+------------\n";

	//std::cout << project.Nodes() << " nodes in " << std::chrono::duration_cast<std::chrono::milliseconds>(project.Duration());
	//if (project.Duration().count())
	//	std::cout << " (" << static_cast<std::size_t>(project.Nodes() / project.Duration().count()) << " N/s)";
	//std::cout << '\n';

	//std::cout << "Tests correct: " << correct << "\n";
	//std::cout << "Tests wrong: " << score_diff.size() - correct << "\n";
	//std::cout << "stddev(score_diff) = " << StandardDeviation(score_diff) << std::endl;

	//std::cout << "TT LookUps: " << tt.LookUpCounter() << " Hits: " << tt.HitCounter() << " Updates: " << tt.UpdateCounter() << std::endl;

	//return 0;
}

