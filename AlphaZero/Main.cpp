#include "Core/Core.h"
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

std::vector<PosScore> LoadTestFiles()
{
	std::vector<GameScore> ret;
	for (int level : {0, 5, 10, 15, 20})
		ret.append_range(LoadGameScoreFile(std::format("G:\\Edax4.4_level_{}_vs_Edax4.4_level_{}_from_e54.gs", level, level)));
	return PosScoreFromGameScores(ret);
}

class FixedDepthPlayer final : public Player
{
	PVS alg;
	int depth;
	float confidence_level;
	std::mt19937_64 rnd_engine;
public:
	FixedDepthPlayer(PVS alg, int depth, float confidence_level, uint64_t seed = std::random_device{}())
		: alg(alg), depth(depth), confidence_level(confidence_level), rnd_engine(seed)
	{}

	Field ChooseMove(const Position& pos) override
	{
		Moves moves = PossibleMoves(pos);
		if (moves.empty())
			return Field::PS;
		if (depth == 0)
		{
			std::size_t rnd = std::uniform_int_distribution<std::size_t>(0, moves.size() - 1)(rnd_engine);
			return moves[rnd];
		}
		else
			return alg.Eval(pos, { -inf_score, +inf_score }, depth, confidence_level).best_move;
	}
};

void Test(const PatternBasedEstimator& estimator)
{
	auto test = LoadTestFiles();
	HT tt{ 10'000'000 };
	PVS alg{ tt, estimator };
	for (int empty_count = 0; empty_count <= 30; empty_count++)
	{
		std::vector<int> diff;
		for (const PosScore& ps : EmptyCountFiltered(test, empty_count))
			diff.push_back(ps.score * 2 - alg.Eval(ps.pos, { -inf_score, +inf_score }, 0, inf).score * 2);
		std::cout << "E " << empty_count << " " << StandardDeviation(diff) << std::endl;
	}
}

int main()
{
	LoggingTimer timer;

	timer.Start("Starting positions");
	auto starting_pos = UniqueChildren(Position::Start(), 50);
	timer.Stop();

	PatternBasedEstimator estimator(/*stage_size*/ 5, pattern::edax);
	HT tt{ 100'000'000 };
	PVS alg{ tt, estimator };
	std::vector<Position> train;

	for (int iteration = 1; iteration <= 10; iteration++)
	{
		tt.clear();
		int depth = (iteration == 1 ? 0 : 5);
		FixedDepthPlayer player(alg, depth, inf, /*seed*/ 18 + iteration);

		timer.Start("Playing games");
		train.append_range(Positions(PlayedGamesFrom(player, player, Sample(1'000'000, starting_pos, /*seed*/ 10 + iteration))));
		timer.Stop();

		timer.Start("Improving score estimator");
		ImproveScoreEstimator(estimator, train, 5, inf, 10);
		timer.Stop();
		 
		//tt.clear();
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

		Serialize(estimator, "G:\\Reversi2\\iteration" + std::to_string(iteration) + ".model");
		Test(estimator);
	}
	return 0;
}