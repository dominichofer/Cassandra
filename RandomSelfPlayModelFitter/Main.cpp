#include "Core/Core.h"
#include "CoreIO/CoreIO.h"
#include "Pattern/Pattern.h"
#include "PatternIO/PatternIO.h"
#include "PatternFit/PatternFit.h"
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

void PrintTimestamp(std::string text = "")
{
	fmt::print("[{0:%F %T}] {1}\n", std::chrono::system_clock::now(), text);
}

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

double StandardDeviation(const AAMSSE& model, const std::vector<PosScore>& reference)
{
	std::vector<int> diff;
	for (const auto& ps : reference)
		diff.push_back(std::round(model.Score(ps.pos)) - ps.score);
	return StandardDeviation(diff);
}

int main(int argc, char* argv[])
{
	std::vector<PosScore> ref = ParsePosScoreFile(R"(G:\Reversi2\Edax4.4_level_20_vs_Edax4.4_level_20_from_e54.pos)");

	int start_pos_pool_empty_count = 50;
	int train_sample_size = 100'000;
	int accuracy_sample_size = 0;
	int eval_intensity = 5;
	int self_play_intensity = 5;
	int accuracy_max_depth = 12;
	int stage_size = 5;
	std::vector<BitBoard> pattern = pattern::edax;

	PrintTimestamp("Start");
	auto start_pos_pool = UniqueChildren(Position::Start(), start_pos_pool_empty_count);
	PrintTimestamp("UniqueChildren");

	std::vector<Position> train_pos, accuracy_pos;
	AAMSSE model;
	for (int it = 1; it <= 100; it++)
	{
		std::vector<Position> a, b;
		if (it == 1)
		{
			a = Positions(RandomGamesFrom(Sample(train_sample_size, start_pos_pool)));
			b = Positions(RandomGamesFrom(Sample(accuracy_sample_size, start_pos_pool)));
		}
		else
		{
			HT tt{ 10'000'000 };
			PVS alg{ tt, model };
			FixedDepthPlayer player(alg, self_play_intensity);
			a = Positions(SelfPlayedGamesFrom(player, Sample(train_sample_size, start_pos_pool)));
			b = Positions(SelfPlayedGamesFrom(player, Sample(accuracy_sample_size, start_pos_pool)));
		}
		train_pos.insert(train_pos.end(), a.begin(), a.end());
		accuracy_pos.insert(accuracy_pos.end(), b.begin(), b.end());
		PrintTimestamp("Self-play");
		
		double R_sq;
		std::tie(model, R_sq) = CreateAAMSSE(stage_size, pattern, train_pos, accuracy_pos, eval_intensity, accuracy_max_depth);
		PrintTimestamp("Fitting");

		Serialize(model, fmt::format(R"(G:\Reversi2\iteration{}.model)", it));
		fmt::print("Iteration {}, stddev {}, R_sq {}\n", it, 2 * StandardDeviation(model, ref), R_sq);
	}

	PrintTimestamp();
	fmt::print("End\n");
	return 0;
}
