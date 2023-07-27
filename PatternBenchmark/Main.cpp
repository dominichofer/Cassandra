#include "benchmark/benchmark.h"
#include "Pattern/Pattern.h"

ScoreEstimator CreateScoreEstimator(std::vector<uint64_t> pattern)
{
	return ScoreEstimator(pattern);
}
ScoreEstimator CreateScoreEstimator(uint64_t pattern)
{
	return CreateScoreEstimator(std::vector{ pattern });
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

void MultiStageScoreEstimator_Edax(benchmark::State& state)
{
	auto evaluator = MultiStageScoreEstimator(/*stage_size*/ 5, pattern::edax);
	Position pos = RandomPosition();

	for (auto _ : state)
	{
		auto value = evaluator.Eval(pos);
		benchmark::DoNotOptimize(value);
	}
	state.SetItemsProcessed(state.iterations());
}
BENCHMARK(MultiStageScoreEstimator_Edax);

BENCHMARK_MAIN();