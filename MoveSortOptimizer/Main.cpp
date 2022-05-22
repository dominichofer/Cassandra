#include "Math/Math.h"
#include "Game/Game.h"
#include "GameIO/GameIO.h"
#include "IO/IO.h"
#include "PatternIO/PatternIO.h"
#include <iostream>
#include <random>
#include <valarray>

std::valarray<float> RandomPoint(std::vector<std::pair<float, float>> range)
{
	static std::mt19937_64 rnd_engine{ std::random_device{}() };
	static std::uniform_real_distribution<float> dist(-1.0f, +1.0f); // upper is not included
	std::valarray<float> p(range.size());
	for (int i = 0; i < range.size(); i++)
		p[i] = std::uniform_real_distribution<float>{ range[i].first, range[i].second }(rnd_engine); // upper is not included
	return p;
}

std::valarray<float> RandomPointOnSphere(int dims)
{
	static std::mt19937_64 rnd_engine{ std::random_device{}() };
	static std::exponential_distribution<float> dist(1);
	static std::discrete_distribution<int> dichotron({1, 1});
	std::valarray<float> p(dims);
	float n = 0.0f;
	while (n == 0.0f)
	{
		for (int i = 0; i < dims; i++)
			p[i] = dichotron(rnd_engine) ? dist(rnd_engine) : -dist(rnd_engine);
		n = norm(p);
	}
	return p / n;
}

//inline CUDA_CALLABLE int DoubleCornerPopcount(const BitBoard& b) noexcept { return popcount(b) + popcount(b & BitBoard::Corners()); }
//inline CUDA_CALLABLE int DoubleCornerPopcount(const Moves& m) noexcept { return m.size() + (m & BitBoard::Corners()).size(); }


float Eval(const Position& pos, const std::valarray<float>& weights)
{
	BitBoard P = pos.Player();
	BitBoard O = pos.Opponent();
	BitBoard E = pos.Empties();
	BitBoard pm = PossibleMoves(pos);
	BitBoard se = StableEdges(pos);
	BitBoard P9 = EightNeighboursAndSelf(P);
	BitBoard O9 = EightNeighboursAndSelf(O);
	BitBoard E9 = EightNeighboursAndSelf(E);

	int i = 0;
	float sum = 0;

	// 10'858
	//sum += DoubleCornerPopcount(PotentialMoves(pos)) << 5;
	//sum += popcount(EightNeighboursAndSelf(pos.Empties()) & pos.Opponent()) << 6;
	//sum += DoubleCornerPopcount(PossibleMoves(pos)) << 15;
	//sum -= DoubleCornerPopcount(StableEdges(pos) & pos.Opponent()) << 11;

	//sum += (weights[i++] * popcount(P & O9));
	sum += (weights[i++] * popcount(P & E9));
	//sum += (weights[i++] * popcount(O & P9));
	sum += (weights[i++] * popcount(O & E9)); // <--
	//sum += (weights[i++] * popcount(E & P9));
	sum += (weights[i++] * popcount(E & O9)); // <--
	//sum += (weights[i++] * popcount(P9 & O9));
	sum += (weights[i++] * popcount(P9 & E9));
	sum += (weights[i++] * popcount(O9 & E9));

	sum += (weights[i++] * DoubleCornerPopcount(pm));
	//sum += (weights[i++] * popcount(pm & P9));
	//sum += (weights[i++] * popcount(pm & O9));

	//sum += (weights[i++] * popcount(se));
	//sum += (weights[i++] * popcount(se & P));
	sum += (weights[i++] * popcount(se & O));
	return sum;
}

Field MoveWithMaxEval(const Position& pos, const std::valarray<float>& weights)
{
	float max_eval = std::numeric_limits<float>::lowest();
	Field max_eval_move = Field::invalid;
	for (Field move : PossibleMoves(pos))
	{
		float eval = Eval(Play(pos, move), weights);
		if (eval > max_eval)
		{
			max_eval = eval;
			max_eval_move = move;
		}
	}
	return max_eval_move;
}

float Cost(const std::vector<AllMovePuzzle>& data, const std::valarray<float>& weights)
{
	float cost = 0.0;
	#pragma omp parallel for reduction(+:cost)
	for (int64_t i = 0; i < static_cast<int64_t>(data.size()); i++)
	{
		const AllMovePuzzle& puzzle = data[i];
		Field best_eval_move = MoveWithMaxEval(puzzle.pos, weights);

		Intensity max_intensity = Intensity::Exact();
		//Intensity max_intensity = puzzle.MaxSolvedIntensity().value();
		std::vector<AllMovePuzzle::Task::SubTask> move_result = puzzle.ResultOf(max_intensity);
		std::ranges::sort(move_result, {}, &AllMovePuzzle::Task::SubTask::result);
		auto it = std::ranges::find(move_result, best_eval_move, &AllMovePuzzle::Task::SubTask::move);
		int index = std::ranges::distance(move_result.begin(), it);
		cost += std::min(index, 2);
	}
	return cost + std::abs(weights).sum();
}

std::valarray<float> Optimize(std::valarray<float> weights, const std::vector<AllMovePuzzle>& data)
{
	float best_cost = Cost(data, weights);
	for (int j = 0; j < 5; j++)
	{
		bool improved = false;
		for (int d = 0; d < weights.size(); d++)
		{
			auto new_weights = weights;
			for (float value = -1.0f; value <= +1.0f; value += 0.01f)
			{
				new_weights[d] = value;
				auto normalized_new_weights = new_weights / norm(new_weights);
				float cost = Cost(data, normalized_new_weights);
				if (cost < best_cost)
				{
					best_cost = cost;
					weights = normalized_new_weights;
					improved = true;
				}
			}
		}
		if (not improved)
			break;
	}
	return weights;
}

//int main()
//{
//	std::vector<Position> positions;
//	for (int e = 0; e < 50; e++)
//	{
//		auto novum = generate_n_unique(std::execution::par, 10'000, PosGen::RandomlyPlayed(/*empty_count*/ 20));
//		positions.insert(positions.end(), novum.begin(), novum.end());
//	}
//	DenseMatrix<int> data(2 * 9, positions.size());
//	for (int i = 0; i < positions.size(); i++)
//	{
//		Position pos = positions[i];
//		auto P = pos.Player();
//		auto O = pos.Opponent();
//		auto E = pos.Empties();
//		auto pm = PossibleMoves(pos);
//		auto P9 = EightNeighboursAndSelf(P);
//		auto O9 = EightNeighboursAndSelf(O);
//		auto E9 = EightNeighboursAndSelf(E);
//
//		int j = 0;
//		data(j++, i) = popcount(P & O9);
//		data(j++, i) = popcount(P & E9);
//		data(j++, i) = popcount(O & P9);
//		data(j++, i) = popcount(O & E9);
//		data(j++, i) = popcount(E & P9); // popcount(E & P9) == popcount(O | P9)
//		data(j++, i) = popcount(E & O9); // popcount(E & O9) == popcount(P | O9)
//		data(j++, i) = popcount(P9 & O9);
//		data(j++, i) = popcount(P9 & E9);
//		data(j++, i) = popcount(O9 & E9);
//
//		data(j++, i) = popcount(P | O9);
//		data(j++, i) = popcount(P | E9);
//		data(j++, i) = popcount(O | P9);
//		data(j++, i) = popcount(O | E9);
//		data(j++, i) = popcount(E | P9);
//		data(j++, i) = popcount(E | O9);
//		data(j++, i) = popcount(P9 | O9);
//		data(j++, i) = popcount(P9 | E9);
//		data(j++, i) = popcount(O9 | E9);
//	}
//	std::cout << to_string(Correlation(data));
//}

int main()
{
	HT tt(100'000'000);
	AAGLEM model = Deserialize<AAGLEM>(R"(G:\Reversi\d5_d5_model_9.mdl)");
	IDAB<PVS> alg{ tt, model };

	DB<AllMovePuzzle> db = Deserialize<DB<AllMovePuzzle>>(R"(G:\Reversi\d5_d5_model_9_e20to23_d5_d5_10k.db)");
	//std::vector<Position> all = AllUnique(Position::Start(), 50);
	//auto player1 = FixedDepthPlayer(alg, 5);
	//auto player2 = FixedDepthPlayer(alg, 5);
	//db.Add({}, generate_n_unique(std::execution::par, 10'000, PosGen::Played(player1, player2, /*empty_count*/ 21, all)));
	//db.Add({}, generate_n_unique(std::execution::par, 10'000, PosGen::Played(player1, player2, /*empty_count*/ 22, all)));
	//db.Add({}, generate_n_unique(std::execution::par, 10'000, PosGen::Played(player1, player2, /*empty_count*/ 23, all)));
	//for (AllMovePuzzle& p : db)
	//	p.insert(Intensity::Exact());

	//auto start = std::chrono::high_resolution_clock::now();
	//Process(std::execution::par, db, [&alg](AllMovePuzzle& p) { p.Solve(alg); });
	//auto stop = std::chrono::high_resolution_clock::now();
	//std::cout << "solving: " << std::chrono::round<std::chrono::seconds>(stop - start) << std::endl;
	//Serialize(db, R"(G:\Reversi\d5_d5_model_9_e20to23_d5_d5_10k.db)");

	std::vector<AllMovePuzzle> data = db | ranges::views::transform([](const auto& x) { return static_cast<AllMovePuzzle>(x); }) | ranges::to_vector;

	int dims = 7;
	float best_cost = std::numeric_limits<float>::max() / 2.0;

	for (int i = 0; i < 1'000; i++)
	{
		std::valarray<float> weights = RandomPointOnSphere(dims);
		float cost = Cost(data, weights);
		if (cost < best_cost)
		{
			best_cost = cost;
			fmt::print("{} {}\n", cost, weights);
		}
	}
	fmt::print("Warm up complete!\n");
	while (true)
	{
		//std::vector<std::pair<float, float>> range{ {0.0f, 0.2f}, { -0.2f, 0.0f }, { 0.1f, 0.4f }, { -0.1f, 0.0f }, { 0.0f, 0.1f }, { -0.2f,0.0f }, { 0.3f, 0.7f }, { -0.6f, -0.0f }, {+2.0f, +2.001f}, {-1.0f, -0.2f} };
		std::valarray<float> weights = RandomPoint({ {-0.6f, -0.2f}, {0.0f, 0.1f}, {-0.6f, -0.4f}, {-0.2f, 0.0f}, {0.0f, 0.4f}, {0.2f, 0.6f}, {-0.7f, -0.5f}/*, {-0.6f, -0.3f}*/ });
		//std::valarray<float> weights = RandomPointOnSphere(dims);
		//weights = { 0.06, -0.075, 0.2, 0.005, 0.02, -0.06, 0.65, -0.8, 2, -0.6 };
		//weights = Optimize(weights, data);

		float cost = Cost(data, weights);
		if (cost < best_cost * 1.001)
		{
			//weights = Optimize(weights, data);
			//cost = Cost(data, weights);
			fmt::print("{} {}\n", cost, weights);
		}
		if (cost < best_cost)
		{
			best_cost = cost;
			fmt::print("{} {}\n", cost, weights);
		}
	}
	return 0;
}