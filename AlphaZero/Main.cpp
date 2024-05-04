#include "Base/Base.h"
#include "IO/IO.h"
#include "Pattern/Pattern.h"
#include "PatternFit/PatternFit.h"
#include "Search/Search.h"
#include <format>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>
#include <random>

std::vector<ScoredPosition> LoadTestData()
{
	std::vector<ScoredPosition> ret;
	for (int level : {1, 5, 10, 15, 20})
	{
		auto file = std::format("C:\\Users\\Dominic\\source\\repos\\python-reversi\\data\\Edax4.4_selfplay_level_{}_from_e54.gs", level);
		ret.append_range(ScoredPositions(LoadScoredGameFile(file)));
	}
	auto file = "C:\\Users\\Dominic\\source\\repos\\python-reversi\\data\\random_selfplay_from_e54.gs";
	ret.append_range(ScoredPositions(LoadScoredGameFile(file)));
	return ret;
}

void Test(const PatternBasedEstimator& estimator)
{
	auto test = LoadTestData();
	RAM_HashTable tt{ 10'000'000 };
	PVS alg{ tt, estimator };
	std::vector<int> all_diff;
	for (int empty_count = 0; empty_count <= 30; empty_count++)
	{
		std::vector<int> diff;
		for (const ScoredPosition& ps : EmptyCountFiltered(test, empty_count))
		{
			auto eval = alg.Eval(ps.pos, 0).window.lower;
			auto d = ps.score * 2 - eval * 2;
			diff.push_back(d);
			all_diff.push_back(d);
		}
		//std::cout << "E " << empty_count << " " << StandardDeviation(diff) << std::endl;
	}
	std::cout << StandardDeviation(all_diff) << std::endl;
}

class FixedDepthPlayer final : public Player
{
	Algorithm& alg;
	Intensity intensity;
public:
	FixedDepthPlayer(Algorithm& alg, Intensity intensity)
		: alg(alg), intensity(intensity)
	{}

	Field ChooseMove(const Position& pos) override
	{
		Moves moves = PossibleMoves(pos);
		if (moves.empty())
			return Field::PS;
		return alg.Eval(pos, intensity).best_move;
	}
};

int main()
{
	const int stage_size = 5;
	const int play_depth = 5;
	const int eval_depth = 5;
	const int sample_size = 100'000;
	std::mt19937_64 rnd(181086);
	LoggingTimer timer;

	timer.Start();
	auto start_pos = UniqueChildren(Position::Start(), /*empty_count*/ 50);
	timer.Stop(std::format("{} unique starting positions", start_pos.size()));

	PatternBasedEstimator estimator(stage_size, pattern::edax);
	RAM_HashTable tt{ 100'000'000 };
	PVS alg{ tt, estimator };
	std::vector<ScoredPosition> train;

	for (int iteration = 1; iteration <= 100; iteration++)
	{
		alg.Clear();

		std::unique_ptr<Player> player;
		if (iteration == 1)
			player = std::make_unique<RandomPlayer>(rnd());
		else
			player = std::make_unique<FixedDepthPlayer>(alg, play_depth);

		timer.Start();
		std::vector<Game> new_games = PlayedGames(*player, *player, Sample(sample_size, start_pos, rnd()));
		//timer.Stop("Playing games");

		train.append_range(ScoredPositions(new_games));

		//timer.Start();
		EvaluateIteratively(estimator, train, eval_depth, 10, /*reevaluate*/ false);
		timer.Stop("Improving score estimator");
		 
		//tt.Clear();
		//timer.Start("Improving accuracy estimator");
		//std::vector<PositionMultiDepthScore> accuracy_pos;
		//for (int e : { 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45 })
		//{
		//	Log("Starting eval of e" + std::to_string(e));
		//	int start = accuracy_pos.size();
		//	for (const Position& pos : Sample(100, EmptyCountFiltered(accuracy_train, e)))
		//		accuracy_pos.push_back(pos);
		//	#pragma omp parallel for
		//	for (int i = start; i < accuracy_pos.size(); i++)
		//	{
		//		const Position& pos = accuracy_pos[i].pos;
		//		if (e <= 20)
		//		{
		//			for (int d = 0; d <= e - 10; d++)
		//				accuracy_pos[i].score_of_depth[d] = alg.Eval(pos, d).Score();
		//			accuracy_pos[i].score_of_depth[e] = alg.Eval(pos, e).Score();
		//		}
		//		else
		//		{
		//			for (int d = 0; d <= 14; d++)
		//				accuracy_pos[i].score_of_depth[d] = alg.Eval(pos, d).Score();
		//		}
		//	}
		//	Log("Done");
		//}

		//using ModelInput = std::vector<int>;
		//std::map<ModelInput, std::vector<int>> score_diffs;

		//for (const PositionMultiDepthScore& datum : accuracy_pos)
		//	for (int D = 0; D < datum.score_of_depth.size(); D++)
		//		if (datum.score_of_depth[D] != undefined_score)
		//			for (int d = 0; d < D; d++)
		//				if (datum.score_of_depth[d] != undefined_score)
		//					score_diffs[{ D, d, datum.pos.EmptyCount() }].push_back(datum.score_of_depth[D] - datum.score_of_depth[d]);

		//std::vector<ModelInput> support;
		//std::vector<int> values;
		//for (auto&& [s, d] : score_diffs)
		//{
		//	support.push_back(s);
		//	values.push_back(StandardDeviation(d));
		//	std::cout << s[0] << " " << s[1] << " " << s[2] << " " << StandardDeviation(d) << std::endl;
		//}

		Serialize(estimator, "G:\\Cassandra\\iteration" + std::to_string(iteration) + ".model");
		Test(estimator);
	}
	return 0;
}