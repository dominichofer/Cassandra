#include "Core/Core.h"
#include "CoreIO/CoreIO.h"
#include "Pattern/Pattern.h"
#include "PatternIO/PatternIO.h"
#include "PatternFit/PatternFit.h"
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

template <typename T>
std::vector<T> Sample(int size, const std::set<T>& pool)
{
	std::mt19937_64 rnd_engine(std::random_device{}());

	std::vector<T> samples;
	samples.reserve(size);
	std::ranges::sample(
		pool,
		std::back_inserter(samples),
		size, rnd_engine);
	return samples;
}

template <typename T>
std::vector<T> Sample(int size, const std::vector<T>& pool)
{
	std::mt19937_64 rnd_engine(std::random_device{}());

	std::vector<T> samples;
	samples.reserve(size);
	std::ranges::sample(
		pool,
		std::back_inserter(samples),
		size, rnd_engine);
	return samples;
}

std::vector<Position> Filtered(const std::vector<Position>& input, int min_empty_count, int max_empty_count)
{
	std::vector<Position> output;
	for (Position pos : input)
		if (min_empty_count <= pos.EmptyCount() and pos.EmptyCount() <= max_empty_count)
			output.push_back(pos);
	return output;
}

std::vector<Position> Filtered(const std::vector<Position>& input, int empty_count)
{
	return Filtered(input, empty_count, empty_count);
}

void PrintHelp()
{
	std::cout
		<< "Outputs 'n' self played games starting from unique positions with 'e' empty fields.\n"
		<< "   -e <int>        Number of empty fields.\n"
		<< "   -n <int>        Number of games to play.\n"
		<< "   -m <string>     Model name.\n"
		<< "   -h              Prints this help."
		<< std::endl;
}

struct AlphaZero
{
	int iteration = 0;

	int unique_empty_count;

	std::string pattern_name;

	Intensity self_play_intensity;

	int train_size;
	Intensity train_eval;
	int exact_blocks;

	std::string save_location;

	int af_size = 0;
	int af_eval_all_depth_till;
	int af_eval_max_depth;

	std::set<Position> start_pos;

	std::vector<Game> train_games;
	std::vector<Game> af_games;

	AAGLEM model;

	AlphaZero(
		int unique_empty_count,
		std::vector<BitBoard> pattern, std::string pattern_name, int block_size,
		Intensity self_play_intensity,
		int train_size, Intensity train_eval, int exact_blocks,
		std::string save_location,
		int af_size = 0, int af_eval_all_depth_till = 0, int af_eval_max_depth = 0)
		: unique_empty_count(unique_empty_count)
		, pattern_name(pattern_name)
		, self_play_intensity(self_play_intensity)
		, train_size(train_size)
		, train_eval(train_eval)
		, exact_blocks(exact_blocks)
		, save_location(save_location)
		, af_size(af_size)
		, af_eval_all_depth_till(af_eval_all_depth_till)
		, af_eval_max_depth(af_eval_max_depth)
		, model(pattern, block_size)
	{
		start_pos = UniqueChildren(Position::Start(), unique_empty_count);
	}

	void Add(const std::vector<Game>& games)
	{
		auto split_point = games.begin() + train_size;

		//if (iteration > 3)
		//{
		//	train_games = Sample(3 * train_size, train_games);
		//	af_games = Sample(3 * af_size, af_games);
		//}
		train_games.insert(train_games.end(), games.begin(), split_point);
		af_games.insert(af_games.end(), split_point, games.end());
		//train_games = std::vector<Game>(games.begin(), split_point);
		//af_games = std::vector<Game>(split_point, games.end());
	}

	void AddRandomGames()
	{
		Add(RandomGamesFrom(Sample(train_size + af_size, start_pos)));
	}

	void AddSelfPlayedGames()
	{
		HT tt{ 10'000'000 };
		PVS alg{ tt, model };
		FixedDepthPlayer player(alg, self_play_intensity);

		Add(SelfPlayedGamesFrom(player, Sample(train_size + af_size, start_pos)));
	}

	double Fit()
	{
		return ::Fit(model, train_games, exact_blocks, train_eval, af_games, af_eval_all_depth_till, af_eval_max_depth);
	}

	void SerializeModel()
	{
		Serialize(model, fmt::format(R"({}\{}.model)", save_location, to_string()));
	}

	auto Iterate()
	{
		std::chrono::high_resolution_clock::time_point start, stop;

		start = std::chrono::high_resolution_clock::now();
		{
			if (iteration == 0)
				AddRandomGames();
			else
				AddSelfPlayedGames();
		}
		stop = std::chrono::high_resolution_clock::now();
		std::chrono::seconds play = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

		start = std::chrono::high_resolution_clock::now();
		{
			Fit();
		}
		stop = std::chrono::high_resolution_clock::now();
		std::chrono::seconds fit = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

		SerializeModel();

		return std::make_tuple(iteration++, play, fit);
	}

	std::string to_string() const
	{
		return fmt::format("pattern_{}_bs_{}_spd{}_train_{}_d{}_exact{}_accfit_{}_{}_{}_it{}",
			pattern_name, model.BlockSize(),
			::to_string(self_play_intensity),
			train_size, ::to_string(train_eval), exact_blocks,
			af_size, af_eval_all_depth_till, af_eval_max_depth,
			iteration
			);
	}
};

double StandardDeviation(const AAGLEM& model, const std::vector<PosScore>& reference)
{
	std::vector<int> diff;
	for (const auto& ps : reference)
		diff.push_back(std::round(model.Eval(ps.pos)) - ps.score);
	return StandardDeviation(diff);
}

void Iterate(std::chrono::seconds duration, AlphaZero az, const std::vector<PosScore>& reference)
{
	auto start = std::chrono::high_resolution_clock::now();
	while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start) < duration)
	{
		auto[it, play, fit] = az.Iterate();

		fmt::print("Iteration {} {} {} {} {:.2f}\n", it, play, fit,
			std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start),
			2 * StandardDeviation(az.model, reference));
	}
}

int main(int argc, char* argv[])
{
	std::vector<PosScore> ref = ParsePosScoreFile(R"(G:\Reversi2\Edax4.4_level_20_vs_Edax4.4_level_20_from_e54.pos)");

	int start_pos_pool_empty_count = 50;
	int sampel_size = 100'000;
	int block_size = 5;
	int blocks = 10;
	int eval_intensity = 5;
	int self_play_intensity = 5;
	AAGLEM model(pattern::edax, 5);

	HT tt{ 10'000'000 };
	PVS alg{ tt, model };
	FixedDepthPlayer player(alg, self_play_intensity);

	std::set<Position> start_pos_pool = UniqueChildren(Position::Start(), start_pos_pool_empty_count);
	
	auto samle_pos = Sample(sampel_size, start_pos_pool);
	auto games = RandomGamesFrom(samle_pos);
	auto pos = Positions(games);
	auto score = std::vector<float>(pos.size(), undefined_score);

	for (int it = 1; it <= 100; it++)
	{
		for (int block = 0; block < blocks; block++)
		{
			#pragma omp parallel for
			for (int i = 0; i < static_cast<int>(pos.size()); i++)
				if (block * block_size <= pos[i].EmptyCount() and pos[i].EmptyCount() < (block + 1) * block_size)
					score[i] = alg.Eval(pos[i], eval_intensity).score;

			std::vector<Position> block_pos;
			std::vector<float> block_score;
			for (std::size_t i = 0; i < pos.size(); i++)
				if (block * block_size <= pos[i].EmptyCount() and pos[i].EmptyCount() < (block + 1) * block_size)
				{
					block_pos.push_back(pos[i]);
					block_score.push_back(score[i]);
				}

			Fit(model.Evaluator(block), block_pos, block_score);

			alg.clear();
			std::cout << 2 * StandardDeviation(model, ref) << std::endl;
		}
		Serialize(model, fmt::format(R"(G:\Reversi2\it{}.model)", it));
		std::cout << "---" << std::endl;
		auto new_samle_pos = Sample(sampel_size, start_pos_pool);
		auto new_games = SelfPlayedGamesFrom(player, new_samle_pos);
		auto new_pos = Positions(new_games);
		pos.insert(pos.end(), new_pos.begin(), new_pos.end());
		score.reserve(pos.size());
	}

	return 0;

	using namespace std::chrono_literals;

	//auto uniques = UniqueChildren(Position::Start(), 50);
	//AAGLEM model;

	//for (int i = 5; i <= 20; i++)
	//{
	//	HT tt{ 10'000'000 };
	//	PVS alg{ tt, model };
	//	FixedDepthPlayer player(alg, i);

	//	auto start = std::chrono::high_resolution_clock::now();

	//	std::fstream stream(fmt::format(R"(G:\Reversi2\555_34_{}.games)", i), std::ios::out);
	//	for (const Game& game : SelfPlayedGamesFrom(player, Sample(1'000, uniques)))
	//		stream << to_string(game) << '\n';
	//	auto stop = std::chrono::high_resolution_clock::now();
	//	std::cout << i << " " << std::chrono::duration_cast<std::chrono::seconds>(stop - start) << std::endl;
	//}

	//AAGLEM model = Deserialize<AAGLEM>(R"(G:\Reversi2\pattern_edax_bs_5_spd5_train_50000_d5_exact0_accfit_0_0_0_it0.model)");
	//std::vector<int> diff;
	//for (const auto& ps : ref)
	//{
	//	int d = std::round(model.Eval(ps.pos));
	//	std::cout << d << " " << ps.score << std::endl;
	//	diff.push_back(d - ps.score);
	//}

	//std::cout << StandardDeviation(diff);
	//return 0;

	// How much data to use?

	//Iterate(3600s, AlphaZero(50, pattern::edax, "edax", /*block_size*/ 1, /*self_play_intensity*/ 3, 200'000, /*train_eval*/ 3, /*exact_blocks*/ 0, R"(G:\Reversi2\)"), ref);
	//Iterate(3600s, AlphaZero(50, pattern::edax, "edax", /*block_size*/ 1, /*self_play_intensity*/ 3, 500'000, /*train_eval*/ 3, /*exact_blocks*/ 0, R"(G:\Reversi2\)"), ref);

	//Iterate(3600s, AlphaZero(50, pattern::edax, "edax", /*block_size*/ 1, /*self_play_intensity*/ 5, 200'000, /*train_eval*/ 3, /*exact_blocks*/ 0, R"(G:\Reversi2\)"), ref);
	//Iterate(3600s, AlphaZero(50, pattern::edax, "edax", /*block_size*/ 1, /*self_play_intensity*/ 5, 500'000, /*train_eval*/ 3, /*exact_blocks*/ 0, R"(G:\Reversi2\)"), ref);

	//Iterate(3600s, AlphaZero(50, pattern::edax, "edax", /*block_size*/ 1, /*self_play_intensity*/ 3, 200'000, /*train_eval*/ 5, /*exact_blocks*/ 0, R"(G:\Reversi2\)"), ref);
	//Iterate(3600s, AlphaZero(50, pattern::edax, "edax", /*block_size*/ 1, /*self_play_intensity*/ 3, 500'000, /*train_eval*/ 5, /*exact_blocks*/ 0, R"(G:\Reversi2\)"), ref);

	//Iterate(3600s, AlphaZero(50, pattern::edax, "edax", /*block_size*/ 1, /*self_play_intensity*/ 5, 200'000, /*train_eval*/ 5, /*exact_blocks*/ 0, R"(G:\Reversi2\)"), ref);
	//Iterate(3600s, AlphaZero(50, pattern::edax, "edax", /*block_size*/ 1, /*self_play_intensity*/ 5, 500'000, /*train_eval*/ 5, /*exact_blocks*/ 0, R"(G:\Reversi2\)"), ref);

	//Iterate(3600s, AlphaZero(50, pattern::edax, "edax", /*block_size*/ 3, /*self_play_intensity*/ 3, 100'000, /*train_eval*/ 3, /*exact_blocks*/ 0, R"(G:\Reversi2\)"), ref);
	//Iterate(3600s, AlphaZero(50, pattern::edax, "edax", /*block_size*/ 3, /*self_play_intensity*/ 3, 200'000, /*train_eval*/ 3, /*exact_blocks*/ 0, R"(G:\Reversi2\)"), ref);

	//Iterate(3600s, AlphaZero(50, pattern::edax, "edax", /*block_size*/ 3, /*self_play_intensity*/ 5, 100'000, /*train_eval*/ 3, /*exact_blocks*/ 0, R"(G:\Reversi2\)"), ref);
	//Iterate(3600s, AlphaZero(50, pattern::edax, "edax", /*block_size*/ 3, /*self_play_intensity*/ 5, 200'000, /*train_eval*/ 3, /*exact_blocks*/ 0, R"(G:\Reversi2\)"), ref);

	//Iterate(3600s, AlphaZero(50, pattern::edax, "edax", /*block_size*/ 3, /*self_play_intensity*/ 3, 100'000, /*train_eval*/ 5, /*exact_blocks*/ 0, R"(G:\Reversi2\)"), ref);
	//Iterate(3600s, AlphaZero(50, pattern::edax, "edax", /*block_size*/ 3, /*self_play_intensity*/ 3, 200'000, /*train_eval*/ 5, /*exact_blocks*/ 0, R"(G:\Reversi2\)"), ref);

	//Iterate(3600s, AlphaZero(50, pattern::edax, "edax", /*block_size*/ 3, /*self_play_intensity*/ 5, 100'000, /*train_eval*/ 5, /*exact_blocks*/ 0, R"(G:\Reversi2\)"), ref); // Best
	//Iterate(3600s, AlphaZero(50, pattern::edax, "edax", /*block_size*/ 3, /*self_play_intensity*/ 5, 200'000, /*train_eval*/ 5, /*exact_blocks*/ 0, R"(G:\Reversi2\)"), ref);

	Iterate(36000s, AlphaZero(50, pattern::edax, "edax", /*block_size*/ 5, /*self_play_intensity*/ 5, 100'000, /*train_eval*/ 5, /*exact_blocks*/ 0, R"(G:\Reversi2\)"), ref);
	//Iteration 0 0s 265s 265s 7.40
	//Iteration 1 40938s 480s 41684s 6.11

	//Iterate(3600s, AlphaZero(50, pattern::logistello, "logistello", /*block_size*/ 5, /*self_play_intensity*/ 5, 100'000, /*train_eval*/ 5, /*exact_blocks*/ 0, R"(G:\Reversi2\)"), ref);

	// Edax 5.49
	//Iteration 0 0s 70s 70s 7.67
	//Iteration 1 36s 109s 217s 6.45
	//Iteration 2 35s 163s 416s 6.29
	//Iteration 3 35s 224s 676s 6.23
	//Iteration 4 38s 294s 1009s 6.21
	//Iteration 5 38s 349s 1397s 6.18
	//Iteration 6 38s 414s 1850s 6.16
	//Iteration 7 38s 480s 2369s 6.16
	//Iteration 8 38s 543s 2951s 6.15
	//Iteration 9 38s 608s 3598s 6.15
	//Iteration 10 38s 673s 4310s 6.13
	//Iteration 11 38s 739s 5088s 6.12
	//Iteration 12 38s 807s 5934s 6.12
	//Iteration 13 39s 884s 6858s 6.11
	//Iteration 14 39s 957s 7855s 6.11
	//Iteration 15 38s 1010s 8904s 6.11
	//Iteration 16 38s 1078s 10020s 6.11
	//Iteration 17 38s 1140s 11200s 6.11
	//Iteration 18 39s 1209s 12448s 6.11
	//Iteration 19 38s 1278s 13765s 6.11
	//Iteration 20 38s 1341s 15145s 6.11
	//Iteration 21 38s 1405s 16590s 6.10
	//Iteration 22 38s 1478s 18107s 6.10
	//Iteration 23 38s 1541s 19687s 6.10
	//Iteration 24 38s 1614s 21340s 6.10
	//Iteration 25 38s 1681s 23061s 6.10
	//Iteration 26 38s 1751s 24851s 6.10
	//Iteration 27 38s 1817s 26707s 6.10
	//Iteration 28 38s 1889s 28636s 6.10
	//Iteration 29 38s 1954s 30629s 6.10
	//Iteration 30 38s 2027s 32695s 6.10
	//Iteration 31 38s 2097s 34831s 6.10
	//Iteration 32 38s 2307s 37178s 6.10
	//Iteration 0 0s 131s 132s 7.47
	//Iteration 1 80s 242s 455s 6.32
	//Iteration 2 78s 365s 898s 6.19
	//Iteration 3 79s 492s 1470s 6.16
	//Iteration 4 78s 624s 2173s 6.14
	//Iteration 5 81s 749s 3003s 6.12
	//Iteration 6 78s 886s 3970s 6.11
	//Iteration 7 79s 1024s 5074s 6.11
	//Iteration 8 80s 1160s 6316s 6.11
	//Iteration 9 79s 1293s 7689s 6.11
	//Iteration 10 79s 1434s 9203s 6.10
	//Iteration 11 80s 1577s 10861s 6.10
	//Iteration 12 79s 1723s 12665s 6.09
	//Iteration 13 81s 1866s 14612s 6.09
	//Iteration 14 81s 2010s 16704s 6.10
	//Iteration 15 79s 2144s 18928s 6.10
	//Iteration 16 79s 2288s 21296s 6.09
	//Iteration 17 81s 2436s 23814s 6.09
	//Iteration 18 79s 2582s 26476s 6.09
	//Iteration 19 83s 2729s 29289s 6.09
	//Iteration 20 79s 2849s 32219s 6.09
	//Iteration 21 80s 2995s 35295s 6.09
	//Iteration 22 80s 3135s 38511s 6.09

	//int unique_empty_count = 50;

	//auto pattern = pattern::edax;
	//int block_size = 5;

	//Intensity self_play_intensity{ 10 };

	//int train_size = 50'000;
	//Intensity train_eval{ 5 };
	//int exact_blocks = 0;

	//int af_size = 100;
	//int af_eval_all_depth_till = 16;
	//int af_eval_max_depth = 8;

	////std::string model_name;
	////for (int i = 0; i < argc; i++)
	////{
	////	if (std::string(argv[i]) == "-e") unique_empty_count = std::stoi(argv[++i]);
	////	else if (std::string(argv[i]) == "-n") train_size = std::stoi(argv[++i]);
	////	else if (std::string(argv[i]) == "-m") model_name = argv[++i];
	////	else if (std::string(argv[i]) == "-h") { PrintHelp(); return 0; }
	////}

	//std::set<Position> unique_chilren = UniqueChildren(Position::Start(), unique_empty_count);

	//auto start = std::chrono::high_resolution_clock::now();
	//std::vector<Game> games = RandomGamesFrom(Sample(train_size + af_size, unique_chilren));
	//auto split_point = games.begin() + train_size;
	//std::vector<Game> train_games(games.begin(), split_point);
	//std::vector<Game> af_games(split_point, games.end());
	//AAGLEM model(pattern, block_size);
	//double R_sq = Fit(model, train_games, exact_blocks, train_eval, af_games, af_eval_all_depth_till, af_eval_max_depth);
	//auto stop = std::chrono::high_resolution_clock::now();
	//std::cout << "Iteration 0, R_sq = " << R_sq << " : " << std::chrono::duration_cast<std::chrono::seconds>(stop - start) << std::endl;
	//Serialize(model, fmt::format(R"(G:\Reversi2\Edax_pattern_{}_{}_{}_games_{}_{}_alpha_zero_iteration_0.model)", block_size, self_play_intensity.depth, train_size, train_eval.depth, exact_blocks));

	//for (int i = 1; i <= 100; i++)
	//{
	//	HT tt{ 10'000'000 };
	//	IDAB<PVS> alg{ tt, model };
	//	FixedDepthPlayer player(alg, self_play_intensity);

	//	auto start = std::chrono::high_resolution_clock::now();
	//	std::vector<Game> new_games = SelfPlayedGamesFrom(player, Sample(train_size + af_size, unique_chilren));
	//	auto split_point = new_games.begin() + train_size;
	//	train_games.insert(train_games.end(), new_games.begin(), split_point);
	//	af_games.insert(af_games.end(), split_point, new_games.end());
	//	auto stop = std::chrono::high_resolution_clock::now();
	//	auto self_play = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

	//	start = std::chrono::high_resolution_clock::now();
	//	double R_sq = Fit(model, train_games, exact_blocks, train_eval, af_games, af_eval_all_depth_till, af_eval_max_depth);
	//	stop = std::chrono::high_resolution_clock::now();
	//	auto fit = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

	//	std::cout << "Iteration " << i << ", R_sq = " << R_sq << " : " << self_play << " : " << fit << std::endl;
	//	Serialize(model, fmt::format(R"(G:\Reversi2\Edax_pattern_{}_{}_{}_games_{}_{}_alpha_zero_iteration_{}.model)", block_size, self_play_intensity.depth, train_size, train_eval.depth, exact_blocks, i));
	//}
	//AAGLEM model2(pattern::logistello, /*block_size*/ 5);
	//Fit(model2, games, train_size, exact_blocks, train_eval, af_size, af_eval_all_depth_till, af_eval_max_depth);
	//Serialize(model2, fmt::format(R"(G:\Reversi2\rnd_{}_logistello.model)", train_size));

	//AAGLEM model3(pattern::cassandra, /*block_size*/ 5);
	//Fit(model3, games, train_size, exact_blocks, train_eval, af_size, af_eval_all_depth_till, af_eval_max_depth);
	//Serialize(model3, fmt::format(R"(G:\Reversi2\rnd_{}_cassandra.model)", train_size));

	return 0;
}
