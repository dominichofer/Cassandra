#include "benchmark/benchmark.h"
#include "Pattern/Pattern.h"

ScoreEstimator CreateScoreEstimator(std::vector<uint64_t> pattern)
{
	std::size_t count = ConfigurationsOfPattern(pattern);
	std::vector<float> w(count, 0);
	return ScoreEstimator(pattern, w);
}
ScoreEstimator CreateScoreEstimator(uint64_t pattern)
{
	return CreateScoreEstimator(std::vector{ pattern });
}

MultiStageScoreEstimator CreateMSSE(int stage_size, std::vector<uint64_t> pattern)
{
	int stages = static_cast<int>(std::ceil(65.0 / stage_size));
	std::size_t count = ConfigurationsOfPattern(pattern) * stages;
	std::vector<float> w(count, 0);
	return MultiStageScoreEstimator(stage_size, pattern, w);
}

void PatternEvalH(benchmark::State& state)
{
	auto evaluator = CreateScoreEstimator(pattern::L0);
	Position pos = RandomPosition();

	for (auto _ : state)
	{
		auto value = evaluator.Eval(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PatternEvalH);

void PatternEvalD(benchmark::State& state)
{
	auto evaluator = CreateScoreEstimator(pattern::D7);
	Position pos = RandomPosition();

	for (auto _ : state)
	{
		auto value = evaluator.Eval(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PatternEvalD);

void PatternEvalA(benchmark::State& state)
{
	auto evaluator = CreateScoreEstimator(pattern::B5);
	Position pos = RandomPosition();

	for (auto _ : state)
	{
		auto value = evaluator.Eval(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PatternEvalA);

void PatternEvalHDA(benchmark::State& state)
{
	auto evaluator = CreateScoreEstimator({ pattern::L0, pattern::D7, pattern::B5 });
	Position pos = RandomPosition();

	for (auto _ : state)
	{
		auto value = evaluator.Eval(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PatternEvalHDA);

void PatternLogistello(benchmark::State& state)
{
	auto evaluator = CreateScoreEstimator(pattern::logistello);
	Position pos = RandomPosition();

	for (auto _ : state)
	{
		auto value = evaluator.Eval(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PatternLogistello);

void PatternEdax(benchmark::State& state)
{
	auto evaluator = CreateScoreEstimator(pattern::edax);
	Position pos = RandomPosition();

	for (auto _ : state)
	{
		auto value = evaluator.Eval(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(PatternEdax);

void MSSE_Edax(benchmark::State& state)
{
	auto evaluator = CreateMSSE(/*stage_size*/ 5, pattern::edax);
	Position pos = RandomPosition();

	for (auto _ : state)
	{
		auto value = evaluator.Eval(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(MSSE_Edax);

BENCHMARK_MAIN();